                 

### FFmpeg音视频处理：多媒体应用开发指南

#### 面试题库

**1. FFmpeg 是什么？请简述其核心功能和用途。**

**答案：** FFmpeg 是一个开源的音频/视频处理软件，它提供了完整的音频/视频编码解码工具和库，可以实现音视频文件的播放、录制、转换、剪辑等功能。其核心功能包括：

* 视频编解码：支持多种视频编解码格式，如 H.264、H.265、HEVC、VP8、VP9 等。
* 音频编解码：支持多种音频编解码格式，如 AAC、MP3、MP4A、FLAC、VORBIS 等。
* 封装格式转换：支持多种封装格式，如 MP4、AVI、MOV、MKV、WEBM 等。
* 音视频同步：可以实现音视频同步，保证播放时的连贯性。

FFmpeg 主要用途包括：

* 音视频处理：对音视频文件进行剪辑、转换、添加特效等操作。
* 在线媒体流：实现流媒体播放和录制功能。
* 编译工具：为其他软件提供音频/视频处理功能。

**2. 如何使用 FFmpeg 对视频进行剪辑？请举例说明。**

**答案：** 使用 FFmpeg 对视频进行剪辑通常使用 `-ss`（指定开始时间）、`-t`（指定时长）和 `-i`（指定输入文件）参数。以下是一个简单的例子：

```bash
ffmpeg -ss 00:00:10 -t 00:01:00 -i input.mp4 output.mp4
```

这个命令会从输入文件 `input.mp4` 中提取 10 秒到 11 秒的视频片段，输出到 `output.mp4` 文件。

**3. 如何使用 FFmpeg 对视频进行缩放？请举例说明。**

**答案：** 使用 FFmpeg 对视频进行缩放可以使用 `-s`（指定输出尺寸）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -s 1280x720 output.mp4
```

这个命令会将对输入文件 `input.mp4` 进行缩放，输出尺寸为 1280x720，保存到 `output.mp4` 文件。

**4. 如何使用 FFmpeg 对视频进行旋转？请举例说明。**

**答案：** 使用 FFmpeg 对视频进行旋转可以使用 `-filter:v`（指定视频滤镜）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -filter:v "transpose=2" output.mp4
```

这个命令会对输入文件 `input.mp4` 进行 90 度旋转，输出到 `output.mp4` 文件。

**5. 如何使用 FFmpeg 对音频进行裁剪？请举例说明。**

**答案：** 使用 FFmpeg 对音频进行裁剪可以使用 `-ss`（指定开始时间）和 `-t`（指定时长）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp3 -ss 00:00:10 -t 00:01:00 output.mp3
```

这个命令会从输入文件 `input.mp3` 中提取 10 秒到 11 秒的音频片段，输出到 `output.mp3` 文件。

**6. 如何使用 FFmpeg 对音频进行混合？请举例说明。**

**答案：** 使用 FFmpeg 对音频进行混合可以使用 `-filter_complex`（指定视频滤镜）参数。以下是一个简单的例子：

```bash
ffmpeg -i audio1.wav -i audio2.wav -filter_complex "amix=inputs=2:duration=longest" output.wav
```

这个命令会将 `audio1.wav` 和 `audio2.wav` 进行混合，输出到 `output.wav` 文件。

**7. 如何使用 FFmpeg 将视频转换为 GIF？请举例说明。**

**答案：** 使用 FFmpeg 将视频转换为 GIF 可以使用 `-ss`（指定开始时间）、`-t`（指定时长）和 `-f`（指定输出格式）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:01:00 -f gif output.gif
```

这个命令会从输入文件 `input.mp4` 中提取 10 秒到 11 秒的视频片段，转换为 GIF 格式，输出到 `output.gif` 文件。

#### 算法编程题库

**1. 如何使用 FFmpeg 编写一个简单的视频播放器？请给出代码示例。**

**答案：** 使用 FFmpeg 库编写一个简单的视频播放器，需要使用 FFmpeg 的 libavformat 和 libavutil 库。以下是一个简单的示例：

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *input_ctx = NULL;
    AVCodecContext *video_ctx = NULL;
    AVCodecContext *audio_ctx = NULL;
    AVFrame *frame = NULL;
    int video_stream_idx = -1;
    int audio_stream_idx = -1;
    int ret;

    if (avformat_open_input(&input_ctx, argv[1], NULL, NULL) < 0) {
        printf("Could not open input file\n");
        return -1;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        printf("Failed to find stream information\n");
        return -1;
    }

    // Find the video and audio streams
    for (int i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
        } else if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
        }
    }

    if (video_stream_idx == -1 || audio_stream_idx == -1) {
        printf("No video or audio stream found\n");
        return -1;
    }

    // Open the video and audio decoders
    video_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(video_ctx, input_ctx->streams[video_stream_idx]->codecpar);
    if (avcodec_open2(video_ctx, avcodec_find_decoder(input_ctx->streams[video_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open video decoder\n");
        return -1;
    }

    audio_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(audio_ctx, input_ctx->streams[audio_stream_idx]->codecpar);
    if (avcodec_open2(audio_ctx, avcodec_find_decoder(input_ctx->streams[audio_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open audio decoder\n");
        return -1;
    }

    // Read packets and decode frames
    AVPacket packet;
    while (av_read_packet(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_idx) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Display the video frame
            // (This part is platform-specific and depends on the video display library you use)
        } else if (packet.stream_index == audio_stream_idx) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, buffer, &buffer_size, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Play the audio frame
            // (This part is platform-specific and depends on the audio playback library you use)
        }
    }

    // Close the decoders and free resources
    avcodec_close(video_ctx);
    avcodec_close(audio_ctx);
    avformat_close_input(&input_ctx);

    return 0;
}
```

在这个示例中，我们首先打开输入文件并找到视频和音频流。然后，我们创建并打开视频和音频解码器上下文。接着，我们读取输入数据包并对其进行解码，最后显示视频帧和播放音频帧。

请注意，这个示例只是一个框架，具体的视频显示和音频播放部分需要根据你的平台和使用的库进行实现。

**2. 如何使用 FFmpeg 编写一个简单的视频录制程序？请给出代码示例。**

**答案：** 使用 FFmpeg 编写一个简单的视频录制程序，需要使用 FFmpeg 的 libavformat 和 libavutil 库。以下是一个简单的示例：

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *output_ctx = NULL;
    AVStream *video_stream = NULL;
    AVStream *audio_stream = NULL;
    AVCodec *video_codec = NULL;
    AVCodec *audio_codec = NULL;
    AVFrame *video_frame = NULL;
    AVFrame *audio_frame = NULL;
    int ret;

    // Open the output file
    if (avformat_alloc_output_context2(&output_ctx, NULL, "flv", argv[2]) < 0) {
        printf("Could not open output file\n");
        return -1;
    }

    // Add video stream
    video_stream = avformat_new_stream(output_ctx, video_codec);
    video_stream->codec->codec_id = AV_CODEC_ID_H264;
    video_stream->codec->time_base = (AVRational){1, 25};
    video_stream->codec->pix_fmt = AV_PIX_FMT_YUV420P;

    // Add audio stream
    audio_stream = avformat_new_stream(output_ctx, audio_codec);
    audio_stream->codec->codec_id = AV_CODEC_ID_AAC;
    audio_stream->codec->time_base = (AVRational){1, 44100};
    audio_stream->codec->sample_fmt = AV_SAMPLE_FMT_S16;
    audio_stream->codec->channel_layout = AV_CH_LAYOUT_STEREO;

    // Write the stream information to the output file
    if (avformat_write_header(output_ctx, NULL) < 0) {
        printf("Error while writing stream information\n");
        return -1;
    }

    // Allocate frame buffers
    video_frame = av_frame_alloc();
    audio_frame = av_frame_alloc();

    // Read input frames and encode them
    AVPacket packet;
    while (av_read_frame(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream->index) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, video_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Encode video frame
            ret = avcodec_encode_video2(video_ctx, &packet, video_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding video frame\n");
                return -1;
            }

            // Write video frame to the output file
            av_interleave_packet(&packet, video_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing video frame\n");
                return -1;
            }
        } else if (packet.stream_index == audio_stream->index) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, audio_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Encode audio frame
            ret = avcodec_encode_audio2(audio_ctx, &packet, audio_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding audio frame\n");
                return -1;
            }

            // Write audio frame to the output file
            av_interleave_packet(&packet, audio_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing audio frame\n");
                return -1;
            }
        }
    }

    // Free resources
    av_frame_free(&video_frame);
    av_frame_free(&audio_frame);
    avformat_free_context(input_ctx);
    avformat_free_context(output_ctx);

    return 0;
}
```

在这个示例中，我们首先创建输出文件和视频、音频流。然后，我们读取输入数据包，解码视频和音频帧，编码并写入输出文件。

请注意，这个示例只是一个框架，具体的视频解码和音频解码部分需要根据你的平台和使用的库进行实现。

#### 详尽答案解析说明

在本节中，我们将对上述面试题和算法编程题的答案进行详细解析，并提供必要的背景知识和实用技巧。

**1. FFmpeg 是什么？请简述其核心功能和用途。**

**解析：** FFmpeg 是一个开源的音频/视频处理软件，它提供了完整的音频/视频编码解码工具和库，可以实现音视频文件的播放、录制、转换、剪辑等功能。其核心功能包括：

- 视频编解码：支持多种视频编解码格式，如 H.264、H.265、HEVC、VP8、VP9 等。
- 音频编解码：支持多种音频编解码格式，如 AAC、MP3、MP4A、FLAC、VORBIS 等。
- 封装格式转换：支持多种封装格式，如 MP4、AVI、MOV、MKV、WEBM 等。
- 音视频同步：可以实现音视频同步，保证播放时的连贯性。

FFmpeg 主要用途包括：

- 音视频处理：对音视频文件进行剪辑、转换、添加特效等操作。
- 在线媒体流：实现流媒体播放和录制功能。
- 编译工具：为其他软件提供音频/视频处理功能。

**2. 如何使用 FFmpeg 对视频进行剪辑？请举例说明。**

**解析：** 使用 FFmpeg 对视频进行剪辑通常使用 `-ss`（指定开始时间）、`-t`（指定时长）和 `-i`（指定输入文件）参数。以下是一个简单的例子：

```bash
ffmpeg -ss 00:00:10 -t 00:01:00 -i input.mp4 output.mp4
```

这个命令会从输入文件 `input.mp4` 中提取 10 秒到 11 秒的视频片段，输出到 `output.mp4` 文件。

- `-ss 00:00:10`：指定剪辑的开始时间为 10 秒。
- `-t 00:01:00`：指定剪辑的时长为 1 分钟（60 秒）。
- `-i input.mp4`：指定输入文件为 `input.mp4`。

**3. 如何使用 FFmpeg 对视频进行缩放？请举例说明。**

**解析：** 使用 FFmpeg 对视频进行缩放可以使用 `-s`（指定输出尺寸）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -s 1280x720 output.mp4
```

这个命令将对输入文件 `input.mp4` 进行缩放，输出尺寸为 1280x720，保存到 `output.mp4` 文件。

- `-s 1280x720`：指定输出视频的尺寸为 1280x720。

**4. 如何使用 FFmpeg 对视频进行旋转？请举例说明。**

**解析：** 使用 FFmpeg 对视频进行旋转可以使用 `-filter:v`（指定视频滤镜）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -filter:v "transpose=2" output.mp4
```

这个命令会对输入文件 `input.mp4` 进行 90 度旋转，输出到 `output.mp4` 文件。

- `-filter:v "transpose=2"`：指定视频滤镜为 `transpose=2`，表示 90 度旋转。

**5. 如何使用 FFmpeg 对音频进行裁剪？请举例说明。**

**解析：** 使用 FFmpeg 对音频进行裁剪可以使用 `-ss`（指定开始时间）和 `-t`（指定时长）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp3 -ss 00:00:10 -t 00:01:00 output.mp3
```

这个命令会从输入文件 `input.mp3` 中提取 10 秒到 11 秒的音频片段，输出到 `output.mp3` 文件。

- `-ss 00:00:10`：指定裁剪的开始时间为 10 秒。
- `-t 00:01:00`：指定裁剪的时长为 1 分钟（60 秒）。

**6. 如何使用 FFmpeg 对音频进行混合？请举例说明。**

**解析：** 使用 FFmpeg 对音频进行混合可以使用 `-filter_complex`（指定视频滤镜）参数。以下是一个简单的例子：

```bash
ffmpeg -i audio1.wav -i audio2.wav -filter_complex "amix=inputs=2:duration=longest" output.wav
```

这个命令会将 `audio1.wav` 和 `audio2.wav` 进行混合，输出到 `output.wav` 文件。

- `-filter_complex "amix=inputs=2:duration=longest"`：指定视频滤镜为 `amix`，表示混合音频。`inputs=2` 表示有两个输入音频流，`duration=longest` 表示混合后的音频时长取最长的一个。

**7. 如何使用 FFmpeg 将视频转换为 GIF？请举例说明。**

**解析：** 使用 FFmpeg 将视频转换为 GIF 可以使用 `-ss`（指定开始时间）、`-t`（指定时长）和 `-f`（指定输出格式）参数。以下是一个简单的例子：

```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:01:00 -f gif output.gif
```

这个命令会从输入文件 `input.mp4` 中提取 10 秒到 11 秒的视频片段，转换为 GIF 格式，输出到 `output.gif` 文件。

- `-i input.mp4`：指定输入文件为 `input.mp4`。
- `-ss 00:00:10`：指定提取的视频片段的开始时间为 10 秒。
- `-t 00:01:00`：指定提取的视频片段的时长为 1 分钟（60 秒）。
- `-f gif`：指定输出格式为 GIF。

**1. 如何使用 FFmpeg 编写一个简单的视频播放器？请给出代码示例。**

**解析：** 使用 FFmpeg 编写一个简单的视频播放器，需要使用 FFmpeg 的 libavformat 和 libavutil 库。以下是一个简单的示例：

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *input_ctx = NULL;
    AVCodecContext *video_ctx = NULL;
    AVCodecContext *audio_ctx = NULL;
    AVFrame *frame = NULL;
    int video_stream_idx = -1;
    int audio_stream_idx = -1;
    int ret;

    if (avformat_open_input(&input_ctx, argv[1], NULL, NULL) < 0) {
        printf("Could not open input file\n");
        return -1;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        printf("Failed to find stream information\n");
        return -1;
    }

    // Find the video and audio streams
    for (int i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
        } else if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
        }
    }

    if (video_stream_idx == -1 || audio_stream_idx == -1) {
        printf("No video or audio stream found\n");
        return -1;
    }

    // Open the video and audio decoders
    video_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(video_ctx, input_ctx->streams[video_stream_idx]->codecpar);
    if (avcodec_open2(video_ctx, avcodec_find_decoder(input_ctx->streams[video_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open video decoder\n");
        return -1;
    }

    audio_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(audio_ctx, input_ctx->streams[audio_stream_idx]->codecpar);
    if (avcodec_open2(audio_ctx, avcodec_find_decoder(input_ctx->streams[audio_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open audio decoder\n");
        return -1;
    }

    // Read packets and decode frames
    AVPacket packet;
    while (av_read_packet(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_idx) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Display the video frame
            // (This part is platform-specific and depends on the video display library you use)
        } else if (packet.stream_index == audio_stream_idx) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, buffer, &buffer_size, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Play the audio frame
            // (This part is platform-specific and depends on the audio playback library you use)
        }
    }

    // Close the decoders and free resources
    avcodec_close(video_ctx);
    avcodec_close(audio_ctx);
    avformat_close_input(&input_ctx);

    return 0;
}
```

在这个示例中，我们首先打开输入文件并找到视频和音频流。然后，我们创建并打开视频和音频解码器上下文。接着，我们读取输入数据包并对其进行解码，最后显示视频帧和播放音频帧。

请注意，这个示例只是一个框架，具体的视频显示和音频播放部分需要根据你的平台和使用的库进行实现。

**2. 如何使用 FFmpeg 编写一个简单的视频录制程序？请给出代码示例。**

**解析：** 使用 FFmpeg 编写一个简单的视频录制程序，需要使用 FFmpeg 的 libavformat 和 libavutil 库。以下是一个简单的示例：

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *output_ctx = NULL;
    AVStream *video_stream = NULL;
    AVStream *audio_stream = NULL;
    AVCodec *video_codec = NULL;
    AVCodec *audio_codec = NULL;
    AVFrame *video_frame = NULL;
    AVFrame *audio_frame = NULL;
    int ret;

    // Open the output file
    if (avformat_alloc_output_context2(&output_ctx, NULL, "flv", argv[2]) < 0) {
        printf("Could not open output file\n");
        return -1;
    }

    // Add video stream
    video_stream = avformat_new_stream(output_ctx, video_codec);
    video_stream->codec->codec_id = AV_CODEC_ID_H264;
    video_stream->codec->time_base = (AVRational){1, 25};
    video_stream->codec->pix_fmt = AV_PIX_FMT_YUV420P;

    // Add audio stream
    audio_stream = avformat_new_stream(output_ctx, audio_codec);
    audio_stream->codec->codec_id = AV_CODEC_ID_AAC;
    audio_stream->codec->time_base = (AVRational){1, 44100};
    audio_stream->codec->sample_fmt = AV_SAMPLE_FMT_S16;
    audio_stream->codec->channel_layout = AV_CH_LAYOUT_STEREO;

    // Write the stream information to the output file
    if (avformat_write_header(output_ctx, NULL) < 0) {
        printf("Error while writing stream information\n");
        return -1;
    }

    // Allocate frame buffers
    video_frame = av_frame_alloc();
    audio_frame = av_frame_alloc();

    // Read input frames and encode them
    AVPacket packet;
    while (av_read_frame(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream->index) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, video_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Encode video frame
            ret = avcodec_encode_video2(video_ctx, &packet, video_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding video frame\n");
                return -1;
            }

            // Write video frame to the output file
            av_interleave_packet(&packet, video_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing video frame\n");
                return -1;
            }
        } else if (packet.stream_index == audio_stream->index) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, audio_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Encode audio frame
            ret = avcodec_encode_audio2(audio_ctx, &packet, audio_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding audio frame\n");
                return -1;
            }

            // Write audio frame to the output file
            av_interleave_packet(&packet, audio_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing audio frame\n");
                return -1;
;
            }
        }
    }

    // Free resources
    av_frame_free(&video_frame);
    av_frame_free(&audio_frame);
    avformat_free_context(input_ctx);
    avformat_free_context(output_ctx);

    return 0;
}
```

在这个示例中，我们首先创建输出文件和视频、音频流。然后，我们读取输入数据包，解码视频和音频帧，编码并写入输出文件。

请注意，这个示例只是一个框架，具体的视频解码和音频解码部分需要根据你的平台和使用的库进行实现。此外，这个示例没有处理输入流的参数（如分辨率、帧率等），在实际应用中需要根据输入流的信息进行相应的配置。

**源代码实例：**

以下是上述两个示例的完整源代码：

**视频播放器示例：**

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *input_ctx = NULL;
    AVCodecContext *video_ctx = NULL;
    AVCodecContext *audio_ctx = NULL;
    AVFrame *frame = NULL;
    int video_stream_idx = -1;
    int audio_stream_idx = -1;
    int ret;

    if (avformat_open_input(&input_ctx, argv[1], NULL, NULL) < 0) {
        printf("Could not open input file\n");
        return -1;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        printf("Failed to find stream information\n");
        return -1;
    }

    // Find the video and audio streams
    for (int i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
        } else if (input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
        }
    }

    if (video_stream_idx == -1 || audio_stream_idx == -1) {
        printf("No video or audio stream found\n");
        return -1;
    }

    // Open the video and audio decoders
    video_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(video_ctx, input_ctx->streams[video_stream_idx]->codecpar);
    if (avcodec_open2(video_ctx, avcodec_find_decoder(input_ctx->streams[video_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open video decoder\n");
        return -1;
    }

    audio_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(audio_ctx, input_ctx->streams[audio_stream_idx]->codecpar);
    if (avcodec_open2(audio_ctx, avcodec_find_decoder(input_ctx->streams[audio_stream_idx]->codecpar->codec_id), NULL) < 0) {
        printf("Could not open audio decoder\n");
        return -1;
    }

    // Read packets and decode frames
    AVPacket packet;
    while (av_read_packet(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_idx) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Display the video frame
            // (This part is platform-specific and depends on the video display library you use)
        } else if (packet.stream_index == audio_stream_idx) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, buffer, &buffer_size, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Play the audio frame
            // (This part is platform-specific and depends on the audio playback library you use)
        }
    }

    // Close the decoders and free resources
    avcodec_close(video_ctx);
    avcodec_close(audio_ctx);
    avformat_close_input(&input_ctx);

    return 0;
}
```

**视频录制程序示例：**

```c
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *output_ctx = NULL;
    AVStream *video_stream = NULL;
    AVStream *audio_stream = NULL;
    AVCodec *video_codec = NULL;
    AVCodec *audio_codec = NULL;
    AVFrame *video_frame = NULL;
    AVFrame *audio_frame = NULL;
    int ret;

    // Open the output file
    if (avformat_alloc_output_context2(&output_ctx, NULL, "flv", argv[2]) < 0) {
        printf("Could not open output file\n");
        return -1;
    }

    // Add video stream
    video_stream = avformat_new_stream(output_ctx, video_codec);
    video_stream->codec->codec_id = AV_CODEC_ID_H264;
    video_stream->codec->time_base = (AVRational){1, 25};
    video_stream->codec->pix_fmt = AV_PIX_FMT_YUV420P;

    // Add audio stream
    audio_stream = avformat_new_stream(output_ctx, audio_codec);
    audio_stream->codec->codec_id = AV_CODEC_ID_AAC;
    audio_stream->codec->time_base = (AVRational){1, 44100};
    audio_stream->codec->sample_fmt = AV_SAMPLE_FMT_S16;
    audio_stream->codec->channel_layout = AV_CH_LAYOUT_STEREO;

    // Write the stream information to the output file
    if (avformat_write_header(output_ctx, NULL) < 0) {
        printf("Error while writing stream information\n");
        return -1;
    }

    // Allocate frame buffers
    video_frame = av_frame_alloc();
    audio_frame = av_frame_alloc();

    // Read input frames and encode them
    AVPacket packet;
    while (av_read_frame(input_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream->index) {
            // Decode video frame
            ret = avcodec_decode_video4(video_ctx, video_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding video frame\n");
                return -1;
            }

            // Encode video frame
            ret = avcodec_encode_video2(video_ctx, &packet, video_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding video frame\n");
                return -1;
            }

            // Write video frame to the output file
            av_interleave_packet(&packet, video_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing video frame\n");
                return -1;
            }
        } else if (packet.stream_index == audio_stream->index) {
            // Decode audio frame
            ret = avcodec_decode_audio4(audio_ctx, audio_frame, &ret, &packet);
            if (ret < 0) {
                printf("Error while decoding audio frame\n");
                return -1;
            }

            // Encode audio frame
            ret = avcodec_encode_audio2(audio_ctx, &packet, audio_frame, &ret);
            if (ret < 0) {
                printf("Error while encoding audio frame\n");
                return -1;
            }

            // Write audio frame to the output file
            av_interleave_packet(&packet, audio_stream);
            if (av_write_frame(output_ctx, &packet) < 0) {
                printf("Error while writing audio frame\n");
                return -1;
            }
        }
    }

    // Free resources
    av_frame_free(&video_frame);
    av_frame_free(&audio_frame);
    avformat_free_context(input_ctx);
    avformat_free_context(output_ctx);

    return 0;
}
```

这些示例代码展示了如何使用 FFmpeg 库的基本功能来编写一个简单的视频播放器和视频录制程序。在实际开发过程中，你可能需要根据具体需求进行扩展和优化。

**总结：** FFmpeg 是一个功能强大的音频/视频处理工具，它可以用于各种多媒体应用的开发。通过掌握 FFmpeg 的基本用法和技巧，开发者可以轻松实现音视频的剪辑、转换、录制等功能。在本节的面试题和算法编程题中，我们通过具体的示例展示了如何使用 FFmpeg 进行音视频处理，并提供了详尽的答案解析和源代码实例。这些知识对于开发者来说是非常实用的，可以帮助他们在项目中快速解决问题并提高开发效率。

