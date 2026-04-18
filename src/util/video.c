/*
 * src/util/video.c - Video frame loading implementation
 *
 * Loads video frames from regular files (via libavformat/libavcodec) or
 * directories of images (via stb_image). Handles resize to model input
 * size and normalization to [-1, 1].
 *
 * Also exposes two internal helpers via util/video_internal.h:
 *   - sam3_frame_dir_list_{open,free}: sorted image-directory listing.
 *   - sam3_rgb_iter_{open,next,close}: native-resolution RGB24 iterator.
 * Both are reused by tools/cli_track.c for the overlay-video pass.
 *
 * Key types:  sam3_video_frames, sam3_video_type, sam3_frame_dir_list,
 *             sam3_rgb_iter
 * Depends on: util/video.h, util/video_internal.h, util/log.h,
 *             vendor/stb_image.h, vendor/stb_image_resize2.h,
 *             libavformat, libavcodec, libswscale, libavutil
 * Used by:    model/video_session.c, tools/cli_track.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>

#include "util/video.h"
#include "util/video_internal.h"
#include "util/log.h"

#include "vendor/stb_image.h"
#include "vendor/stb_image_resize2.h"

/* libav — suppress warnings in vendored headers */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define MAX_DIR_FRAMES 4096

/* --- filename helpers --- */

static int has_suffix(const char *str, const char *suffix)
{
	size_t slen = strlen(str);
	size_t xlen = strlen(suffix);
	if (slen < xlen)
		return 0;
	const char *tail = str + slen - xlen;
	for (size_t i = 0; i < xlen; i++) {
		if (tolower((unsigned char)tail[i]) !=
		    tolower((unsigned char)suffix[i]))
			return 0;
	}
	return 1;
}

static int is_image_file(const char *name)
{
	return has_suffix(name, ".jpg") || has_suffix(name, ".jpeg") ||
	       has_suffix(name, ".png") || has_suffix(name, ".ppm") ||
	       has_suffix(name, ".bmp");
}

static int parse_numeric_prefix(const char *name, long *val)
{
	char *end;
	long v = strtol(name, &end, 10);
	if (end == name)
		return 0;
	*val = v;
	return 1;
}

static int cmp_filenames(const void *a, const void *b)
{
	const char *sa = *(const char *const *)a;
	const char *sb = *(const char *const *)b;
	long va, vb;
	if (parse_numeric_prefix(sa, &va) && parse_numeric_prefix(sb, &vb)) {
		if (va != vb)
			return (va > vb) - (va < vb);
	}
	return strcmp(sa, sb);
}

/* --- sam3_frame_dir_list (shared with cli_track) --- */

enum sam3_error sam3_frame_dir_list_open(const char *dir_path,
					 struct sam3_frame_dir_list *out)
{
	if (!dir_path || !out)
		return SAM3_EINVAL;

	memset(out, 0, sizeof(*out));
	out->dir_path = dir_path;

	DIR *dir = opendir(dir_path);
	if (!dir) {
		sam3_log_error("cannot open frame directory '%s'", dir_path);
		return SAM3_EIO;
	}

	char **names = NULL;
	int n = 0, cap = 0;
	struct dirent *ent;
	enum sam3_error err = SAM3_OK;

	while ((ent = readdir(dir)) != NULL) {
		if (!is_image_file(ent->d_name))
			continue;
		if (n >= MAX_DIR_FRAMES) {
			sam3_log_warn("frame directory capped at %d files",
				      MAX_DIR_FRAMES);
			break;
		}
		if (n >= cap) {
			cap = cap ? cap * 2 : 64;
			char **tmp = realloc(names,
					     (size_t)cap * sizeof(char *));
			if (!tmp) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			names = tmp;
		}
		names[n] = strdup(ent->d_name);
		if (!names[n]) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		n++;
	}
	closedir(dir);

	if (n == 0) {
		sam3_log_error("no image files in '%s'", dir_path);
		err = SAM3_EIO;
		goto fail;
	}

	qsort(names, (size_t)n, sizeof(char *), cmp_filenames);
	out->names = names;
	out->n = n;
	return SAM3_OK;

fail:
	closedir(dir);
	for (int i = 0; i < n; i++)
		free(names[i]);
	free(names);
	memset(out, 0, sizeof(*out));
	return err;
}

void sam3_frame_dir_list_free(struct sam3_frame_dir_list *list)
{
	if (!list)
		return;
	for (int i = 0; i < list->n; i++)
		free(list->names[i]);
	free(list->names);
	memset(list, 0, sizeof(*list));
}

/* --- make_frame_tensor (shared by dir + libav load paths) --- */

static struct sam3_tensor *make_frame_tensor(const uint8_t *rgb,
					     int w, int h,
					     struct sam3_arena *arena)
{
	struct sam3_tensor *t = sam3_arena_alloc(arena,
						 sizeof(struct sam3_tensor));
	if (!t)
		return NULL;

	memset(t, 0, sizeof(*t));
	t->dtype = SAM3_DTYPE_F32;
	t->n_dims = 3;
	t->dims[0] = 3;
	t->dims[1] = h;
	t->dims[2] = w;
	sam3_tensor_compute_strides(t);
	t->nbytes = (size_t)3 * h * w * sizeof(float);

	float *data = sam3_arena_alloc(arena, t->nbytes);
	if (!data)
		return NULL;
	t->data = data;

	int hw = h * w;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int px = y * w + x;
			const uint8_t *src = rgb + (y * w + x) * 3;
			data[0 * hw + px] = (src[0] / 255.0f - 0.5f) / 0.5f;
			data[1 * hw + px] = (src[1] / 255.0f - 0.5f) / 0.5f;
			data[2 * hw + px] = (src[2] / 255.0f - 0.5f) / 0.5f;
		}
	}
	return t;
}

/* --- frame-directory load (unchanged behavior) --- */

static enum sam3_error load_frame_dir(const char *path, int image_size,
				      struct sam3_video_frames *out,
				      struct sam3_arena *arena)
{
	struct sam3_frame_dir_list list;
	enum sam3_error err = sam3_frame_dir_list_open(path, &list);
	if (err != SAM3_OK)
		return err;

	uint8_t *resized = NULL;

	out->pixels = sam3_arena_alloc(arena,
				       (size_t)list.n *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		err = SAM3_ENOMEM;
		goto done;
	}

	out->n_frames   = 0;
	out->frame_size = image_size;
	out->fps_num    = 0;
	out->fps_den    = 1;

	resized = malloc((size_t)image_size * image_size * 3);
	if (!resized) {
		sam3_log_error("out of memory for resize buffer");
		err = SAM3_ENOMEM;
		goto done;
	}

	for (int i = 0; i < list.n; i++) {
		char fpath[1024];
		int n = snprintf(fpath, sizeof(fpath), "%s/%s",
				 list.dir_path, list.names[i]);
		if (n < 0 || (size_t)n >= sizeof(fpath)) {
			sam3_log_warn("path too long, skipping '%s'",
				      list.names[i]);
			continue;
		}

		int w, h, ch;
		uint8_t *pixels = stbi_load(fpath, &w, &h, &ch, 3);
		if (!pixels) {
			sam3_log_error("failed to load '%s': %s", fpath,
				       stbi_failure_reason());
			err = SAM3_EIO;
			goto done;
		}

		if (out->n_frames == 0) {
			out->orig_width  = w;
			out->orig_height = h;
		}

		uint8_t *r = stbir_resize_uint8_linear(
			pixels, w, h, 0,
			resized, image_size, image_size, 0,
			STBIR_RGB);
		stbi_image_free(pixels);
		if (!r) {
			sam3_log_error("resize failed for '%s'", fpath);
			err = SAM3_EIO;
			goto done;
		}

		struct sam3_tensor *t = make_frame_tensor(resized, image_size,
							  image_size, arena);
		if (!t) {
			sam3_log_error("out of memory encoding frame '%s'",
				       fpath);
			err = SAM3_ENOMEM;
			goto done;
		}
		out->pixels[out->n_frames++] = t;
	}

	sam3_log_info("loaded %d frames from '%s' (%dx%d -> %d)",
		      out->n_frames, path, out->orig_width,
		      out->orig_height, image_size);

done:
	free(resized);
	sam3_frame_dir_list_free(&list);
	return err;
}

/* --- libav helpers (shared by load_libav and sam3_rgb_iter) --- */

struct libav_decoder {
	AVFormatContext  *fmt;
	AVCodecContext   *codec;
	AVFrame          *frame;      /* decoded frame in native pixel format */
	AVFrame          *rgb_frame;  /* RGB24 target (data owned by us) */
	AVPacket         *packet;
	struct SwsContext *sws;
	int               stream_idx;
	int               width;
	int               height;
	AVRational        avg_frame_rate;
};

static void libav_decoder_close(struct libav_decoder *d)
{
	if (!d)
		return;
	if (d->sws)
		sws_freeContext(d->sws);
	if (d->rgb_frame) {
		av_freep(&d->rgb_frame->data[0]);
		av_frame_free(&d->rgb_frame);
	}
	if (d->frame)
		av_frame_free(&d->frame);
	if (d->packet)
		av_packet_free(&d->packet);
	if (d->codec)
		avcodec_free_context(&d->codec);
	if (d->fmt)
		avformat_close_input(&d->fmt);
	memset(d, 0, sizeof(*d));
}

static enum sam3_error libav_decoder_open(struct libav_decoder *d,
					  const char *path)
{
	memset(d, 0, sizeof(*d));

	int rc = avformat_open_input(&d->fmt, path, NULL, NULL);
	if (rc < 0) {
		sam3_log_error("cannot open '%s': %s", path, av_err2str(rc));
		return SAM3_EIO;
	}
	rc = avformat_find_stream_info(d->fmt, NULL);
	if (rc < 0) {
		sam3_log_error("cannot find stream info in '%s': %s",
			       path, av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}

	int stream_idx = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO,
					     -1, -1, NULL, 0);
	if (stream_idx < 0) {
		sam3_log_error("no video stream in '%s'", path);
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	d->stream_idx = stream_idx;
	AVStream *stream = d->fmt->streams[stream_idx];
	d->avg_frame_rate = stream->avg_frame_rate;

	const AVCodec *dec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!dec) {
		sam3_log_error("no decoder for codec id %d",
			       stream->codecpar->codec_id);
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	d->codec = avcodec_alloc_context3(dec);
	if (!d->codec) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	rc = avcodec_parameters_to_context(d->codec, stream->codecpar);
	if (rc < 0) {
		sam3_log_error("avcodec_parameters_to_context: %s",
			       av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	rc = avcodec_open2(d->codec, dec, NULL);
	if (rc < 0) {
		sam3_log_error("avcodec_open2: %s", av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}

	d->width  = d->codec->width;
	d->height = d->codec->height;

	d->frame     = av_frame_alloc();
	d->rgb_frame = av_frame_alloc();
	d->packet    = av_packet_alloc();
	if (!d->frame || !d->rgb_frame || !d->packet) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	d->rgb_frame->format = AV_PIX_FMT_RGB24;
	d->rgb_frame->width  = d->width;
	d->rgb_frame->height = d->height;
	rc = av_image_alloc(d->rgb_frame->data, d->rgb_frame->linesize,
			    d->width, d->height, AV_PIX_FMT_RGB24, 1);
	if (rc < 0) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}

	d->sws = sws_getContext(d->width, d->height, d->codec->pix_fmt,
				d->width, d->height, AV_PIX_FMT_RGB24,
				SWS_BILINEAR, NULL, NULL, NULL);
	if (!d->sws) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	return SAM3_OK;
}

/*
 * libav_decoder_next - Decode the next frame into d->rgb_frame.
 *
 * On success, *out_eof is 0 and d->rgb_frame holds an RGB24 frame of
 * size d->width x d->height. On end-of-stream, *out_eof is 1 and the
 * frame buffer contents are undefined.
 */
static enum sam3_error libav_decoder_next(struct libav_decoder *d,
					  int *out_eof)
{
	*out_eof = 0;
	for (;;) {
		int rc = avcodec_receive_frame(d->codec, d->frame);
		if (rc == 0) {
			sws_scale(d->sws,
				  (const uint8_t * const *)d->frame->data,
				  d->frame->linesize, 0, d->height,
				  d->rgb_frame->data, d->rgb_frame->linesize);
			return SAM3_OK;
		}
		if (rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) {
			sam3_log_error("avcodec_receive_frame: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
		if (rc == AVERROR_EOF) {
			*out_eof = 1;
			return SAM3_OK;
		}
		/* Need more input — read a packet. */
		rc = av_read_frame(d->fmt, d->packet);
		if (rc == AVERROR_EOF) {
			avcodec_send_packet(d->codec, NULL); /* flush */
			continue;
		}
		if (rc < 0) {
			sam3_log_error("av_read_frame: %s", av_err2str(rc));
			return SAM3_EIO;
		}
		if (d->packet->stream_index != d->stream_idx) {
			av_packet_unref(d->packet);
			continue;
		}
		rc = avcodec_send_packet(d->codec, d->packet);
		av_packet_unref(d->packet);
		if (rc < 0 && rc != AVERROR(EAGAIN)) {
			sam3_log_error("avcodec_send_packet: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
	}
}

/* --- model-ready libav load (replaces load_mpeg) --- */

static enum sam3_error load_libav(const char *path, int image_size,
				  struct sam3_video_frames *out,
				  struct sam3_arena *arena)
{
	struct libav_decoder d;
	enum sam3_error err = libav_decoder_open(&d, path);
	if (err != SAM3_OK)
		return err;

	out->orig_width  = d.width;
	out->orig_height = d.height;
	out->frame_size  = image_size;
	out->fps_num     = d.avg_frame_rate.num;
	out->fps_den     = d.avg_frame_rate.den > 0 ? d.avg_frame_rate.den : 1;

	/* First pass: count frames */
	int n_total = 0;
	for (;;) {
		int eof;
		err = libav_decoder_next(&d, &eof);
		if (err != SAM3_OK) {
			libav_decoder_close(&d);
			return err;
		}
		if (eof)
			break;
		n_total++;
	}
	if (n_total == 0) {
		sam3_log_error("no decodable video frames in '%s'", path);
		libav_decoder_close(&d);
		return SAM3_EIO;
	}
	libav_decoder_close(&d);

	/* Reopen for the real pass */
	err = libav_decoder_open(&d, path);
	if (err != SAM3_OK)
		return err;

	out->pixels = sam3_arena_alloc(arena,
				       (size_t)n_total *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		libav_decoder_close(&d);
		return SAM3_ENOMEM;
	}

	uint8_t *resized = malloc((size_t)image_size * image_size * 3);
	if (!resized) {
		libav_decoder_close(&d);
		return SAM3_ENOMEM;
	}

	out->n_frames = 0;
	for (int i = 0; i < n_total; i++) {
		int eof;
		err = libav_decoder_next(&d, &eof);
		if (err != SAM3_OK || eof)
			break;

		const uint8_t *rgb = d.rgb_frame->data[0];
		uint8_t *r = stbir_resize_uint8_linear(
			rgb, d.width, d.height, d.rgb_frame->linesize[0],
			resized, image_size, image_size, 0,
			STBIR_RGB);
		if (!r) {
			sam3_log_error("resize failed for frame %d", i);
			err = SAM3_EIO;
			break;
		}
		struct sam3_tensor *t = make_frame_tensor(resized, image_size,
							  image_size, arena);
		if (!t) {
			err = SAM3_ENOMEM;
			break;
		}
		out->pixels[out->n_frames++] = t;
	}

	free(resized);
	libav_decoder_close(&d);

	if (err == SAM3_OK)
		sam3_log_info("loaded %d frames from '%s' (%dx%d -> %d)",
			      out->n_frames, path, out->orig_width,
			      out->orig_height, image_size);
	return err;
}

/* --- public API --- */

enum sam3_video_type sam3_video_detect_type(const char *path)
{
	if (!path)
		return SAM3_VIDEO_UNKNOWN;

	struct stat st;
	if (stat(path, &st) != 0)
		return SAM3_VIDEO_UNKNOWN;
	if (S_ISDIR(st.st_mode))
		return SAM3_VIDEO_FRAME_DIR;
	if (S_ISREG(st.st_mode))
		return SAM3_VIDEO_FILE;
	return SAM3_VIDEO_UNKNOWN;
}

enum sam3_error sam3_video_load(const char *path, int image_size,
				struct sam3_video_frames *out,
				struct sam3_arena *arena)
{
	if (!path || !out || !arena || image_size <= 0) {
		sam3_log_error("invalid args to sam3_video_load");
		return SAM3_EINVAL;
	}
	memset(out, 0, sizeof(*out));

	enum sam3_video_type vtype = sam3_video_detect_type(path);
	switch (vtype) {
	case SAM3_VIDEO_FRAME_DIR:
		return load_frame_dir(path, image_size, out, arena);
	case SAM3_VIDEO_FILE:
		return load_libav(path, image_size, out, arena);
	default:
		sam3_log_error("unknown video source type for '%s'", path);
		return SAM3_EIO;
	}
}

/* --- sam3_rgb_iter (native-resolution RGB24, for overlay pass 2) --- */

enum sam3_rgb_iter_kind {
	RGB_ITER_LIBAV = 1,
	RGB_ITER_DIR   = 2,
};

struct sam3_rgb_iter {
	enum sam3_rgb_iter_kind  kind;

	/* Libav mode */
	struct libav_decoder     dec;

	/* Dir mode */
	struct sam3_frame_dir_list list;
	int                        dir_cursor;
	uint8_t                   *dir_last_rgb;  /* stb_image-owned */
};

enum sam3_error sam3_rgb_iter_open(const char *path,
				   struct sam3_rgb_iter **out)
{
	if (!path || !out)
		return SAM3_EINVAL;
	*out = NULL;

	struct sam3_rgb_iter *it = calloc(1, sizeof(*it));
	if (!it)
		return SAM3_ENOMEM;

	enum sam3_video_type vt = sam3_video_detect_type(path);
	if (vt == SAM3_VIDEO_FRAME_DIR) {
		enum sam3_error err =
			sam3_frame_dir_list_open(path, &it->list);
		if (err != SAM3_OK) {
			free(it);
			return err;
		}
		it->kind = RGB_ITER_DIR;
	} else if (vt == SAM3_VIDEO_FILE) {
		enum sam3_error err = libav_decoder_open(&it->dec, path);
		if (err != SAM3_OK) {
			free(it);
			return err;
		}
		it->kind = RGB_ITER_LIBAV;
	} else {
		sam3_log_error("cannot iterate '%s' (not a file or dir)",
			       path);
		free(it);
		return SAM3_EIO;
	}
	*out = it;
	return SAM3_OK;
}

enum sam3_error sam3_rgb_iter_next(struct sam3_rgb_iter *it,
				   const uint8_t **out_rgb,
				   int *out_w, int *out_h, int *out_eof)
{
	if (!it || !out_rgb || !out_w || !out_h || !out_eof)
		return SAM3_EINVAL;

	*out_rgb = NULL;
	*out_eof = 0;

	if (it->kind == RGB_ITER_LIBAV) {
		int eof;
		enum sam3_error err = libav_decoder_next(&it->dec, &eof);
		if (err != SAM3_OK)
			return err;
		if (eof) {
			*out_eof = 1;
			return SAM3_OK;
		}
		*out_rgb = it->dec.rgb_frame->data[0];
		*out_w   = it->dec.width;
		*out_h   = it->dec.height;
		return SAM3_OK;
	}

	/* RGB_ITER_DIR */
	if (it->dir_last_rgb) {
		stbi_image_free(it->dir_last_rgb);
		it->dir_last_rgb = NULL;
	}
	if (it->dir_cursor >= it->list.n) {
		*out_eof = 1;
		return SAM3_OK;
	}
	char fpath[1024];
	int n = snprintf(fpath, sizeof(fpath), "%s/%s",
			 it->list.dir_path,
			 it->list.names[it->dir_cursor]);
	if (n < 0 || (size_t)n >= sizeof(fpath)) {
		sam3_log_error("rgb iter: path too long");
		return SAM3_EIO;
	}
	int w, h, ch;
	uint8_t *pixels = stbi_load(fpath, &w, &h, &ch, 3);
	if (!pixels) {
		sam3_log_error("rgb iter: stbi_load '%s': %s", fpath,
			       stbi_failure_reason());
		return SAM3_EIO;
	}
	it->dir_last_rgb = pixels;
	it->dir_cursor++;
	*out_rgb = pixels;
	*out_w = w;
	*out_h = h;
	return SAM3_OK;
}

void sam3_rgb_iter_close(struct sam3_rgb_iter *it)
{
	if (!it)
		return;
	if (it->kind == RGB_ITER_LIBAV) {
		libav_decoder_close(&it->dec);
	} else if (it->kind == RGB_ITER_DIR) {
		if (it->dir_last_rgb)
			stbi_image_free(it->dir_last_rgb);
		sam3_frame_dir_list_free(&it->list);
	}
	free(it);
}

void sam3_rgb_iter_fps(const struct sam3_rgb_iter *it,
		       int *out_num, int *out_den)
{
	if (out_num) *out_num = 0;
	if (out_den) *out_den = 1;
	if (!it)
		return;
	if (it->kind == RGB_ITER_LIBAV) {
		if (out_num) *out_num = it->dec.avg_frame_rate.num;
		if (out_den) *out_den = it->dec.avg_frame_rate.den > 0
			? it->dec.avg_frame_rate.den : 1;
	}
}
