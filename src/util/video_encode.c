/*
 * src/util/video_encode.c - Encoder + overlay helper
 *
 * Implements sam3_overlay_composite (pure C, no libav) and the
 * libav-backed encoder open/write/close.
 *
 * Key types:  sam3_video_encoder
 * Depends on: util/video_encode.h, util/log.h, libavformat,
 *             libavcodec, libswscale, libavutil
 * Used by:    tools/cli_track.c, tests/test_video_encode.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "util/video_encode.h"
#include "util/log.h"

/*
 * Ten visually distinguishable colors. Ordered so cycling mod 10 gives
 * consecutive objects high contrast. Values chosen to be reasonably
 * distinct across common displays.
 */
static const uint8_t palette[10][3] = {
	{ 239,  71, 111 }, /* pink/red */
	{  17, 138, 178 }, /* teal */
	{ 255, 209, 102 }, /* amber */
	{   6, 214, 160 }, /* mint */
	{ 155,  93, 229 }, /* purple */
	{ 251, 133,   0 }, /* orange */
	{  42, 157, 143 }, /* sea */
	{ 231, 111,  81 }, /* coral */
	{  38,  70,  83 }, /* navy */
	{ 244, 162,  97 }, /* tan */
};

void sam3_overlay_composite(uint8_t *rgb, int w, int h,
			    const uint8_t *mask, int mw, int mh,
			    int obj_id, float alpha)
{
	if (!rgb || !mask || w <= 0 || h <= 0 || mw <= 0 || mh <= 0)
		return;
	if (alpha < 0.0f) alpha = 0.0f;
	if (alpha > 1.0f) alpha = 1.0f;

	int idx = obj_id % 10;
	if (idx < 0) idx += 10;
	const uint8_t *c = palette[idx];

	float inv = 1.0f - alpha;
	float cr = c[0] * alpha;
	float cg = c[1] * alpha;
	float cb = c[2] * alpha;

	for (int y = 0; y < h; y++) {
		int my = (int)((long)y * mh / h);
		const uint8_t *mrow = mask + my * mw;
		uint8_t *orow = rgb + (size_t)y * w * 3;
		for (int x = 0; x < w; x++) {
			int mx = (int)((long)x * mw / w);
			if (!mrow[mx])
				continue;
			orow[x * 3 + 0] =
				(uint8_t)(orow[x * 3 + 0] * inv + cr);
			orow[x * 3 + 1] =
				(uint8_t)(orow[x * 3 + 1] * inv + cg);
			orow[x * 3 + 2] =
				(uint8_t)(orow[x * 3 + 2] * inv + cb);
		}
	}
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

struct sam3_video_encoder {
	AVFormatContext *fmt;
	AVStream        *stream;
	AVCodecContext  *codec;
	AVFrame         *yuv_frame;
	AVPacket        *packet;
	struct SwsContext *sws;
	int              width;
	int              height;
	int              fps_num;
	int              fps_den;
	int64_t          next_pts;
	int              header_written;
	int              io_opened;
};

static enum AVCodecID codec_for_path(const char *path)
{
	size_t n = strlen(path);
	const char *ext = NULL;
	for (size_t i = n; i > 0; i--) {
		if (path[i - 1] == '.') { ext = path + i; break; }
	}
	if (!ext) return AV_CODEC_ID_NONE;
	if (strcasecmp(ext, "mp4") == 0 ||
	    strcasecmp(ext, "mov") == 0 ||
	    strcasecmp(ext, "mkv") == 0)
		return AV_CODEC_ID_H264;
	if (strcasecmp(ext, "webm") == 0)
		return AV_CODEC_ID_VP9;
	return AV_CODEC_ID_NONE;
}

enum sam3_error sam3_video_encoder_close(struct sam3_video_encoder *enc)
{
	enum sam3_error ret = SAM3_OK;
	if (!enc)
		return SAM3_OK;

	/* Flush encoder if we opened it */
	if (enc->codec && enc->packet) {
		avcodec_send_frame(enc->codec, NULL);
		for (;;) {
			int rc = avcodec_receive_packet(enc->codec, enc->packet);
			if (rc == AVERROR_EOF || rc == AVERROR(EAGAIN))
				break;
			if (rc < 0) {
				sam3_log_error("encoder flush: %s",
					       av_err2str(rc));
				ret = SAM3_EIO;
				break;
			}
			av_packet_rescale_ts(enc->packet,
					     enc->codec->time_base,
					     enc->stream->time_base);
			enc->packet->stream_index = enc->stream->index;
			int wr = av_interleaved_write_frame(enc->fmt,
							    enc->packet);
			av_packet_unref(enc->packet);
			if (wr < 0) {
				sam3_log_error("interleaved_write_frame: %s",
					       av_err2str(wr));
				ret = SAM3_EIO;
				break;
			}
		}
	}

	if (enc->header_written && enc->fmt) {
		int rc = av_write_trailer(enc->fmt);
		if (rc < 0) {
			sam3_log_error("av_write_trailer: %s",
				       av_err2str(rc));
			ret = SAM3_EIO;
		}
	}

	if (enc->sws)       sws_freeContext(enc->sws);
	if (enc->yuv_frame) av_frame_free(&enc->yuv_frame);
	if (enc->packet)    av_packet_free(&enc->packet);
	if (enc->codec)     avcodec_free_context(&enc->codec);
	if (enc->io_opened && enc->fmt &&
	    !(enc->fmt->oformat->flags & AVFMT_NOFILE))
		avio_closep(&enc->fmt->pb);
	if (enc->fmt)       avformat_free_context(enc->fmt);

	free(enc);
	return ret;
}

enum sam3_error sam3_video_encoder_open(const char *path,
					int width, int height,
					int fps_num, int fps_den,
					struct sam3_video_encoder **out)
{
	if (!path || !out || width <= 0 || height <= 0 ||
	    fps_num <= 0 || fps_den <= 0)
		return SAM3_EINVAL;
	*out = NULL;

	enum AVCodecID codec_id = codec_for_path(path);
	if (codec_id == AV_CODEC_ID_NONE) {
		sam3_log_error("unsupported output extension for '%s' "
			       "(accepted: .mp4/.mov/.mkv/.webm)", path);
		return SAM3_EIO;
	}

	struct sam3_video_encoder *enc = calloc(1, sizeof(*enc));
	if (!enc)
		return SAM3_ENOMEM;
	enc->width   = width;
	enc->height  = height;
	enc->fps_num = fps_num;
	enc->fps_den = fps_den;

	int rc;
	rc = avformat_alloc_output_context2(&enc->fmt, NULL, NULL, path);
	if (rc < 0 || !enc->fmt) {
		sam3_log_error("avformat_alloc_output_context2: %s",
			       av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	const AVCodec *codec = avcodec_find_encoder(codec_id);
	if (!codec) {
		sam3_log_error("encoder not compiled into libav for codec %d "
			       "(rebuild ffmpeg with libx264/libvpx)",
			       (int)codec_id);
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	enc->stream = avformat_new_stream(enc->fmt, NULL);
	if (!enc->stream) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	enc->codec = avcodec_alloc_context3(codec);
	if (!enc->codec) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}
	enc->codec->codec_id   = codec_id;
	enc->codec->width      = width;
	enc->codec->height     = height;
	enc->codec->pix_fmt    = AV_PIX_FMT_YUV420P;
	enc->codec->time_base  = (AVRational){ fps_den, fps_num };
	enc->codec->framerate  = (AVRational){ fps_num, fps_den };
	enc->codec->gop_size   = 12;
	enc->codec->max_b_frames = 0;

	if (enc->fmt->oformat->flags & AVFMT_GLOBALHEADER)
		enc->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	if (codec_id == AV_CODEC_ID_H264) {
		av_opt_set(enc->codec->priv_data, "preset", "medium", 0);
		av_opt_set(enc->codec->priv_data, "crf", "23", 0);
	}

	rc = avcodec_open2(enc->codec, codec, NULL);
	if (rc < 0) {
		sam3_log_error("avcodec_open2: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	rc = avcodec_parameters_from_context(enc->stream->codecpar, enc->codec);
	if (rc < 0) {
		sam3_log_error("avcodec_parameters_from_context: %s",
			       av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}
	enc->stream->time_base = enc->codec->time_base;

	enc->yuv_frame = av_frame_alloc();
	enc->packet    = av_packet_alloc();
	if (!enc->yuv_frame || !enc->packet) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}
	enc->yuv_frame->format = AV_PIX_FMT_YUV420P;
	enc->yuv_frame->width  = width;
	enc->yuv_frame->height = height;
	rc = av_frame_get_buffer(enc->yuv_frame, 32);
	if (rc < 0) {
		sam3_log_error("av_frame_get_buffer: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	enc->sws = sws_getContext(width, height, AV_PIX_FMT_RGB24,
				  width, height, AV_PIX_FMT_YUV420P,
				  SWS_BILINEAR, NULL, NULL, NULL);
	if (!enc->sws) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	if (!(enc->fmt->oformat->flags & AVFMT_NOFILE)) {
		rc = avio_open(&enc->fmt->pb, path, AVIO_FLAG_WRITE);
		if (rc < 0) {
			sam3_log_error("avio_open '%s': %s", path,
				       av_err2str(rc));
			sam3_video_encoder_close(enc);
			return SAM3_EIO;
		}
		enc->io_opened = 1;
	}

	rc = avformat_write_header(enc->fmt, NULL);
	if (rc < 0) {
		sam3_log_error("avformat_write_header: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}
	enc->header_written = 1;

	*out = enc;
	return SAM3_OK;
}

enum sam3_error sam3_video_encoder_write_rgb(struct sam3_video_encoder *enc,
					     const uint8_t *rgb)
{
	if (!enc || !rgb)
		return SAM3_EINVAL;

	int rc = av_frame_make_writable(enc->yuv_frame);
	if (rc < 0) {
		sam3_log_error("av_frame_make_writable: %s", av_err2str(rc));
		return SAM3_EIO;
	}

	const uint8_t *src_slices[4]   = { rgb, NULL, NULL, NULL };
	int            src_strides[4]  = { enc->width * 3, 0, 0, 0 };
	sws_scale(enc->sws, src_slices, src_strides, 0, enc->height,
		  enc->yuv_frame->data, enc->yuv_frame->linesize);

	enc->yuv_frame->pts = enc->next_pts++;

	rc = avcodec_send_frame(enc->codec, enc->yuv_frame);
	if (rc < 0) {
		sam3_log_error("avcodec_send_frame: %s", av_err2str(rc));
		return SAM3_EIO;
	}

	for (;;) {
		rc = avcodec_receive_packet(enc->codec, enc->packet);
		if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF)
			return SAM3_OK;
		if (rc < 0) {
			sam3_log_error("avcodec_receive_packet: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
		av_packet_rescale_ts(enc->packet, enc->codec->time_base,
				     enc->stream->time_base);
		enc->packet->stream_index = enc->stream->index;
		int wr = av_interleaved_write_frame(enc->fmt, enc->packet);
		av_packet_unref(enc->packet);
		if (wr < 0) {
			sam3_log_error("av_interleaved_write_frame: %s",
				       av_err2str(wr));
			return SAM3_EIO;
		}
	}
}
