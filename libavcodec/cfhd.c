/*
 * Copyright (c) 2015 Kieran Kunhya
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * CFHD Video Decoder
 */

#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"

#include "avcodec.h"
#include "bswapdsp.h"
#include "internal.h"
#include "cfhd.h"

static av_cold int cfhd_decode_init(AVCodecContext *avctx)
{
    CFHDContext *s = avctx->priv_data;

    avctx->pix_fmt             = AV_PIX_FMT_YUV422P10;
    avctx->bits_per_raw_sample = 10;
    s->avctx                   = avctx;

    return ff_cfhd_init_vlcs(s);
}

static void init_plane_defaults(CFHDContext *s)
{
    s->subband_num        = 0;
    s->level              = 0;
    s->subband_num_actual = 0;
}

static void init_frame_defaults(CFHDContext *s)
{
    s->bpc               = 10;
    s->channel_cnt       = 4;
    s->subband_cnt       = 10;
    s->channel_num       = 0;
    s->lowpass_precision = 16;
    s->quantisation      = 1;
    s->wavelet_depth     = 3;
    s->pshift            = 1;
    s->codebook          = 0;
    init_plane_defaults(s);
}

static inline int dequant_and_decompand(int level, int quantisation)
{
    int abslevel = abs(level);
    return (abslevel + ((768 * abslevel * abslevel * abslevel) / (255 * 255 * 255))) * FFSIGN(level) * quantisation;
}

static inline void filter(int16_t *output, ptrdiff_t out_stride, int16_t *low, ptrdiff_t low_stride,
                          int16_t *high, ptrdiff_t high_stride, int len)
{
    int32_t tmp;

    int i;
    for (i = 0; i < len; i++) {
        if( i == 0 )
        {
            tmp = (11*low[0*low_stride] - 4*low[1*low_stride] + low[2*low_stride] + 4) >> 3;
            output[(2*i+0)*out_stride] = (tmp + high[0*high_stride]) >> 1;
            tmp = ( 5*low[0*low_stride] + 4*low[1*low_stride] - low[2*low_stride] + 4) >> 3;
            output[(2*i+1)*out_stride] = (tmp - high[0*high_stride]) >> 1;
        }
        else if( i == len-1 )
        {
            tmp = ( 5*low[i*low_stride] + 4*low[(i-1)*low_stride] - low[(i-2)*low_stride] + 4) >> 3;
            output[(2*i+0)*out_stride] = (tmp + high[i*high_stride]) >> 1;
            tmp = (11*low[i*low_stride] - 4*low[(i-1)*low_stride] + low[(i-2)*low_stride] + 4) >> 3;
            output[(2*i+1)*out_stride] = (tmp - high[i*high_stride]) >> 1;
        }
        else
        {
            tmp = (low[(i-1)*low_stride] - low[(i+1)*low_stride] + 4) >> 3;
            output[(2*i+0)*out_stride] = (tmp + low[i*low_stride] + high[i*high_stride]) >> 1;
            tmp = (low[(i+1)*low_stride] - low[(i-1)*low_stride] + 4) >> 3;
            output[(2*i+1)*out_stride] = (tmp + low[i*low_stride] - high[i*high_stride]) >> 1;
        }
    }
}

static void horiz_filter(int16_t *output, int16_t *low, int16_t *high, int width)
{
    filter(output, 1, low, 1, high, 1, width);
}

static void vert_filter(int16_t *output, int out_stride, int16_t *low, int low_stride,
                        int16_t *high, int high_stride, int len)
{
    filter(output, out_stride, low, low_stride, high, high_stride, len);
}

static int cfhd_decode(AVCodecContext *avctx, void *data, int *got_frame,
                       AVPacket *avpkt)
{
    CFHDContext *s = avctx->priv_data;
    uint8_t *bs = avpkt->data;
    int cnt = 0;
    AVFrame *pic = data;
    int ret = 0, i, j;
    int16_t *plane[3] = {NULL};
    int16_t *tmp[3] = {NULL};
    int16_t *subband[3][10] = {{0}};
    int16_t *l_h[3][8];
    int16_t *coeff_data;

    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_x_shift, &s->chroma_y_shift);

    for (i = 0; i < 3; i++) {
        int width = i ? avctx->width >> s->chroma_x_shift : avctx->width;
        int height = i ? avctx->height >> s->chroma_y_shift : avctx->height;
        int stride = FFALIGN(width / 8, 8) * 8;
        int w8, h8, w4, h4, w2, h2;
        height = FFALIGN(height / 8, 2) * 8;
        s->plane[i].width = width;
        s->plane[i].height = height;
        s->plane[i].stride = stride;

        w8 = FFALIGN(s->plane[i].width / 8, 8);
        h8 = FFALIGN(s->plane[i].height / 8, 2);
        w4 = w8 * 2;
        h4 = h8 * 2;
        w2 = w4 * 2;
        h2 = h4 * 2;

        plane[i] = av_malloc(height * stride * sizeof(*plane[i]));
        tmp[i]   = av_malloc(height * stride * sizeof(*tmp[i]));
        if (!plane[i] || !tmp[i]) {
            ret = AVERROR(ENOMEM);
            goto end;
        }

        subband[i][0] = plane[i];
        subband[i][1] = plane[i] + 2 * w8 * h8;
        subband[i][2] = plane[i] + 1 * w8 * h8;
        subband[i][3] = plane[i] + 3 * w8 * h8;
        subband[i][4] = plane[i] + 2 * w4 * h4;
        subband[i][5] = plane[i] + 1 * w4 * h4;
        subband[i][6] = plane[i] + 3 * w4 * h4;
        subband[i][7] = plane[i] + 2 * w2 * h2;
        subband[i][8] = plane[i] + 1 * w2 * h2;
        subband[i][9] = plane[i] + 3 * w2 * h2;

        l_h[i][0] = tmp[i];
        l_h[i][1] = tmp[i] + 2 * w8 * h8;
        //l_h[i][2] = ll2;
        l_h[i][3] = tmp[i];
        l_h[i][4] = tmp[i] + 2 * w4 * h4;
        //l_h[i][5] = ll1;
        l_h[i][6] = tmp[i];
        l_h[i][7] = tmp[i] + 2 * w2 * h2;
    }

    init_frame_defaults(s);

    if ((ret = ff_get_buffer(avctx, pic, 0)) < 0)
        return ret;

    while (cnt < avpkt->size) {
        int16_t tag     = AV_RB16(&bs[cnt]);
        int8_t tag8     = (int8_t)bs[cnt];
        uint16_t abstag = abs(tag);
        int8_t abs_tag8 = abs(tag8);
        uint16_t data   = AV_RB16(&bs[cnt + 2]);
        if (abs_tag8 >= 0x60 && abs_tag8 <= 0x6f) {
            av_log(avctx, AV_LOG_DEBUG, "large len %x \n", AV_RB24(&bs[cnt + 1]));
        } else if (tag == 20) {
            av_log(avctx, AV_LOG_DEBUG, "Width %"PRIu16" %x \n", data, cnt);
            avctx->width = data;
        } else if (tag == 21) {
            av_log(avctx, AV_LOG_DEBUG, "Height %"PRIu16" %x \n", data, cnt);
            avctx->height = data;
        } else if (tag == 101) {
            av_log(avctx, AV_LOG_DEBUG, "Bits per component: %"PRIu16" \n", data);
            s->bpc = data;
        } else if (tag == 12) {
            av_log(avctx, AV_LOG_DEBUG, "Channel Count: %"PRIu16" \n", data);
            s->channel_cnt = data;
            if (data != 3) {
                av_log(avctx, AV_LOG_ERROR, "Channel Count of %"PRIu16" is unsupported\n", data);
                ret = AVERROR_PATCHWELCOME;
                break;
            }
        } else if (tag == 14) {
            av_log(avctx, AV_LOG_DEBUG, "Subband Count: %"PRIu16" \n", data);
            if (data != 10) {
                av_log(avctx, AV_LOG_ERROR, "Subband Count of %"PRIu16" is unsupported\n", data);
                ret = AVERROR_PATCHWELCOME;
                break;
            }
        } else if (tag == 62) {
            s->channel_num = data;
            av_log(avctx, AV_LOG_DEBUG, "Channel number %"PRIu16" \n", data);
            init_plane_defaults(s);
        } else if (tag == 48) {
            if (s->subband_num != 0 && data == 1)  // hack
                s->level++;
            av_log(avctx, AV_LOG_DEBUG, "Subband number %"PRIu16" \n", data);
            s->subband_num = data;
        } else if (tag == 51) {
            av_log(avctx, AV_LOG_DEBUG, "Subband number actual %"PRIu16" \n", data);
            s->subband_num_actual = data;
        } else if (tag == 35)
            av_log(avctx, AV_LOG_DEBUG, "Lowpass precision bits: %"PRIu16" \n", data);
        else if (tag == 53) {
            s->quantisation = data;
            av_log(avctx, AV_LOG_DEBUG, "Quantisation: %"PRIu16" \n", data);
        } else if (tag == 109) {
            s->prescale_shift[0] = (data >> 0) & 0x7;
            s->prescale_shift[1] = (data >> 3) & 0x7;
            s->prescale_shift[2] = (data >> 6) & 0x7;
            av_log(avctx, AV_LOG_DEBUG, "Prescale shift (VC-5): %x \n", data);
        } else if (tag == 27) {
            s->plane[s->channel_num].band[0][0].width  = data;
            s->plane[s->channel_num].band[0][0].stride = data;
            av_log(avctx, AV_LOG_DEBUG, "Lowpass width %"PRIu16" \n", data);
        } else if (tag == 28) {
            s->plane[s->channel_num].band[0][0].height = data;
            av_log(avctx, AV_LOG_DEBUG, "Lowpass height %"PRIu16" \n", data);
        } else if (tag == 1)
            av_log(avctx, AV_LOG_DEBUG, "Sample type? %"PRIu16" \n", data);
        else if (tag == 10) {
            if (data != 0) {
                av_log(avctx, AV_LOG_ERROR, "Transform type of %"PRIu16" is unsupported\n", data);
                ret = AVERROR_PATCHWELCOME;
                break;
            }
            av_log(avctx, AV_LOG_DEBUG, "Transform-type? %"PRIu16" \n", data);
        }
        else if (abstag >= 0x4000 && abstag <= 0x40ff) {
            av_log(avctx, AV_LOG_DEBUG, "Small chunk length %"PRIu16" %s \n", data * 4, tag < 0 ? "optional" : "required");
            cnt += data * 4;
        } else if (tag == 23) {
            av_log(avctx, AV_LOG_DEBUG, "Skip frame \n");
            av_log(avctx, AV_LOG_ERROR, "Skip frame not supported \n");
            ret = AVERROR_PATCHWELCOME;
            break;
        } else if (tag == 2) {
            av_log(avctx, AV_LOG_DEBUG, "tag=2 header - skipping %i tag/value pairs \n", data);
            for (i = 0; i < data + 1; i++) {
                av_log(avctx, AV_LOG_DEBUG, "Tag/Value = %x %x \n", AV_RB16(&bs[cnt]), AV_RB16(&bs[cnt + 2]));
                cnt += 4;
            }
        } else if (tag == 41) {
            s->plane[s->channel_num].band[s->level][s->subband_num].width  = data;
            s->plane[s->channel_num].band[s->level][s->subband_num].stride = FFALIGN(data, 8);
            av_log(avctx, AV_LOG_DEBUG, "Highpass width %i channel %i level %i subband %i \n", data, s->channel_num, s->level, s->subband_num);
        } else if (tag == 42) {
            s->plane[s->channel_num].band[s->level][s->subband_num].height = data;
            av_log(avctx, AV_LOG_DEBUG, "Highpass height %i \n", data);
        } else if (tag == 49) {
            s->plane[s->channel_num].band[s->level][s->subband_num].width  = data;
            s->plane[s->channel_num].band[s->level][s->subband_num].stride = FFALIGN(data, 8);
            av_log(avctx, AV_LOG_DEBUG, "Highpass width2 %i \n", data);
        } else if (tag == 50) {
            s->plane[s->channel_num].band[s->level][s->subband_num].height = data;
            av_log(avctx, AV_LOG_DEBUG, "Highpass height2 %i \n", data);
        } else if (tag == 71) {
            s->codebook = data;
            av_log(avctx, AV_LOG_DEBUG, "Codebook %i \n", s->codebook);
        } else if (tag == 72) {
            s->codebook = data;
            av_log(avctx, AV_LOG_DEBUG, "Other codebook? %i \n", s->codebook);
        } else
            av_log(avctx, AV_LOG_DEBUG,  "Unknown tag %i data %x \n", tag, data);
        cnt += 4;

        coeff_data = subband[s->channel_num][s->subband_num_actual];

        /* Lowpass coefficients */
        if (tag == 4 && data == 0xf0f) {
            int lowpass_height = s->plane[s->channel_num].band[0][0].height;
            int lowpass_width  = s->plane[s->channel_num].band[0][0].width;
            uint16_t coeffs    = 0;
            av_log(avctx, AV_LOG_DEBUG, "Start of lowpass coeffs component %"PRIu16" \n", s->channel_num);
            for (i = 0; i < lowpass_height; i++) {
                for (j = 0; j < lowpass_width; j++) {
                    coeff_data[j] = AV_RB16(&bs[cnt]);

                    coeffs++;
                    cnt += 2;
                }
                coeff_data += lowpass_width;
            }

            /* Copy last line of coefficients if odd height */
            if (lowpass_height & 1) {
                memcpy(&coeff_data[lowpass_height * lowpass_width],
                       &coeff_data[(lowpass_height - 1) * lowpass_width],
                       lowpass_width * sizeof(*coeff_data));
            }

            av_log(avctx, AV_LOG_DEBUG, "Lowpass coefficients %"PRIu16" \n", coeffs);
        }

        if (tag == 55 && s->subband_num_actual != 255) {
            int highpass_height = s->plane[s->channel_num].band[s->level][s->subband_num].height;
            int highpass_stride = s->plane[s->channel_num].band[s->level][s->subband_num].stride;
            int expected = highpass_height * highpass_stride;
            int level, run, coeff;
            int count = 0;

            av_log(avctx, AV_LOG_DEBUG, "Start subband coeffs plane %i level %i codebook %i expected %i \n", s->channel_num, s->level, s->codebook, expected);

            init_get_bits(&s->gb, &bs[cnt], (avpkt->size - cnt) * 8);
            OPEN_READER(re, &s->gb);
            if (!s->codebook) {
                for (;;) {
                    UPDATE_CACHE(re, &s->gb);
                    GET_RL_VLC(level, run, re, &s->gb, s->table_9_rl_vlc,
                               VLC_BITS, 3, 1);

                    /* escape */
                    if (level == 64)
                        break;

                    count += run;

                    if (count > expected)
                        break;

                    coeff = dequant_and_decompand(level, s->quantisation);
                    for (i = 0; i < run; i++)
                        *coeff_data++ = coeff;
                }
            } else {
                for (;;) {
                    UPDATE_CACHE(re, &s->gb);
                    GET_RL_VLC(level, run, re, &s->gb, s->table_18_rl_vlc,
                               VLC_BITS, 3, 1);

                    /* escape */
                    if (level == 255 && run == 2)
                        break;

                    count += run;

                    if (count > expected)
                        break;

                    coeff = dequant_and_decompand(level, s->quantisation);
                    for (i = 0; i < run; i++)
                        *coeff_data++ = coeff;
                }
            }
            CLOSE_READER(re, &s->gb);

            if (count > expected) {
                av_log(avctx, AV_LOG_ERROR, "Escape codeword not found, probably corrupt data");
                break;
            }

            cnt += FFALIGN(FF_CEIL_RSHIFT(get_bits_count(&s->gb), 3), 4);
            av_log(avctx, AV_LOG_DEBUG, "End subband coeffs %i extra %i %p \n", count, count - expected, subband[s->subband_num_actual]);
            s->codebook = 0;

            /* Copy last line of coefficients if odd height */
            if (highpass_height & 1) {
                memcpy(&coeff_data[highpass_height * highpass_stride],
                       &coeff_data[(highpass_height - 1) * highpass_stride],
                       highpass_stride * sizeof(*coeff_data));
            }
        }
    }

    for (int plane = 0; plane < 3 && !ret; plane++) {
        /* level 1 */
        int lowpass_height  = s->plane[plane].band[0][0].height;
        int lowpass_width   = s->plane[plane].band[0][0].width;
        int highpass_stride = s->plane[plane].band[0][1].stride;
        int act_plane = plane == 1 ? 2 : plane == 2 ? 1 : 0;
        int16_t *low, *high, *output, *dst;

        av_log(avctx, AV_LOG_DEBUG, "Decoding level 1 plane %i %i %i %i \n", plane, lowpass_height, lowpass_width, highpass_stride);

        low    = subband[plane][0];
        high   = subband[plane][2];
        output = l_h[plane][0];
        for (i = 0; i < lowpass_width; i++) {
            vert_filter(output, lowpass_width, low, lowpass_width, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        low    = subband[plane][1];
        high   = subband[plane][3];
        output = l_h[plane][1];
        for (i = 0; i < lowpass_width; i++) {
            // note the stride of "low" is highpass_stride
            vert_filter(output, lowpass_width, low, highpass_stride, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        low    = l_h[plane][0];
        high   = l_h[plane][1];
        output = subband[plane][0];
        for (i = 0; i < lowpass_height * 2; i++) {
            horiz_filter(output, low, high, lowpass_width);
            low    += lowpass_width;
            high   += lowpass_width;
            output += lowpass_width * 2;
        }

        /* level 2 */
        lowpass_height  = s->plane[plane].band[1][1].height;
        lowpass_width   = s->plane[plane].band[1][1].width;
        highpass_stride = s->plane[plane].band[1][1].stride;

        av_log(avctx, AV_LOG_DEBUG, "Level 2 plane %i %i %i %i \n", plane, lowpass_height, lowpass_width, highpass_stride);

        low    = subband[plane][0];
        high   = subband[plane][5];
        output = l_h[plane][3];
        for (i = 0; i < lowpass_width; i++) {
            vert_filter(output, lowpass_width, low, lowpass_width, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        low    = subband[plane][4];
        high   = subband[plane][6];
        output = l_h[plane][4];
        for (i = 0; i < lowpass_width; i++) {
            vert_filter(output, lowpass_width, low, highpass_stride, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        low    = l_h[plane][3];
        high   = l_h[plane][4];
        output = subband[plane][0];
        for (i = 0; i < lowpass_height * 2; i++) {
            horiz_filter(output, low, high, lowpass_width);
            low    += lowpass_width;
            high   += lowpass_width;
            output += lowpass_width * 2;
        }

        output = subband[plane][0];
        for (i = 0; i < lowpass_height * 2; i++) {
            for (j = 0; j < lowpass_width * 2; j++)
                output[j] <<= 2;

            output += lowpass_width * 2;
        }

        /* level 3 */
        lowpass_height  = s->plane[plane].band[2][1].height;
        lowpass_width   = s->plane[plane].band[2][1].width;
        highpass_stride = s->plane[plane].band[2][1].stride;

        av_log(avctx, AV_LOG_DEBUG, "Level 3 plane %i %i %i %i \n", plane, lowpass_height, lowpass_width, highpass_stride);

        low    = subband[plane][0];
        high   = subband[plane][8];
        output = l_h[plane][6];
        for (i = 0; i < lowpass_width; i++) {
            vert_filter(output, lowpass_width, low, lowpass_width, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        low    = subband[plane][7];
        high   = subband[plane][9];
        output = l_h[plane][7];
        for (i = 0; i < lowpass_width; i++) {
            vert_filter(output, lowpass_width, low, highpass_stride, high, highpass_stride, lowpass_height);
            low++;
            high++;
            output++;
        }

        dst = (int16_t *)pic->data[act_plane];
        low  = l_h[plane][6];
        high = l_h[plane][7];
        for (i = 0; i < lowpass_height * 2; i++) {
            horiz_filter(dst, low, high, lowpass_width);
            low  += lowpass_width;
            high += lowpass_width;
            dst  += pic->linesize[act_plane] / 2;
        }
    }


end:
    for (i = 0; i < 3; i++) {
        av_freep(&plane[i]);
        av_freep(&tmp[i]);
    }

    if (ret < 0)
        return ret;

    *got_frame = 1;
    return avpkt->size;
}

static av_cold int cfhd_close_decoder(AVCodecContext *avctx)
{
    CFHDContext *s = avctx->priv_data;

    if (!avctx->internal->is_copy) {
        ff_free_vlc(&s->vlc_9);
        ff_free_vlc(&s->vlc_18);
    }

    return 0;
}

AVCodec ff_cfhd_decoder = {
    .name           = "cfhd",
    .long_name      = NULL_IF_CONFIG_SMALL("Cineform HD"),
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_CFHD,
    .priv_data_size = sizeof(CFHDContext),
    .init           = cfhd_decode_init,
    .close          = cfhd_close_decoder,
    .decode         = cfhd_decode,
    .capabilities   = AV_CODEC_CAP_EXPERIMENTAL | AV_CODEC_CAP_DR1 | AV_CODEC_CAP_FRAME_THREADS,
    .caps_internal  = FF_CODEC_CAP_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP,
};
