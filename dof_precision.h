#pragma once
#include <string>

namespace gles {

// Depth of field / bloom post-process shaders (gather, Gaussian blur, composite) declare their texture
// coordinates `varying mediump` and run at `precision mediump float`. Apple's GPUs interpolate those
// at full precision; Adreno and Mali really use 16 bits. A half float has ~11 bits of mantissa, so near
// the right and bottom of the screen a coordinate snaps in steps of 0.0005: that is 0.4 texel of the
// 796x362 blur buffers and 1.5 pixels of a 3176 px wide render. The blur taps (placed between texels to
// get bilinear weights) land unevenly, which shows as streaks and banding in the background blur, and the
// composite's full-resolution lookups snap to a coarse grid. These shaders are rewritten to full precision.
// The shared gather vertex shader (TextureUV) and its other fragment shaders change together, so the
// vertex outputs and fragment inputs of every program keep the same precision.
inline bool dof_chain_shader(const std::string& s) {
    auto has = [&](const char* t) { return s.find(t) != std::string::npos; };
    return has("varying mediump vec4 OffsetUVs") || has("varying mediump vec2 OffsetUVs") ||
           has("varying mediump vec2 SourceTextureUV") || has("varying mediump vec2 TextureUV");
}

inline std::string highp_dof_chain(const std::string& src) {
    if (!dof_chain_shader(src)) return src;
    std::string out;
    out.reserve(src.size() + 64);
    size_t pos = 0;
    while (pos < src.size()) {
        size_t end = src.find('\n', pos);
        end = end == std::string::npos ? src.size() : end + 1;
        std::string line = src.substr(pos, end - pos);
        static const std::string varying = "varying mediump ";
        if (line.compare(0, varying.size(), varying) == 0) line.replace(8, 7, "highp");
        out += line;
        pos = end;
    }
    static const std::string medium = "precision mediump float;";
    size_t at = out.find(medium);
    if (at != std::string::npos) out.replace(at, medium.size(), "precision highp float;\nprecision highp sampler2D;");
    return out;
}

// The Gaussian blur's vertex shader builds each pair of tap coordinates as
//     OffsetUVs[i] = TexCoords0.xyyx + SampleOffsets16[i];     (the pixel shader reads .xy and then .wz)
// which expects the offsets packed as (xA, yA, yB, xB). The engine writes them as (xA, yA, xB, yB), as the
// frame log shows: in the horizontal pass the third value is already an x offset. So the second tap of every
// pair lands on the wrong axis (above the pixel in the horizontal pass, beside it in the vertical pass), and
// the blur turns into a half-strength cross with streaks instead of a Gaussian. Swizzling the offsets in the
// vertex shader to (x, y, w, z) makes the pixel shader's .wz come out as (u + xB, v + yB).
inline std::string fix_blur_offset_order(const std::string& src) {
    static const std::string key = "TexCoords0.xyyx + SampleOffsets";
    if (src.find(key) == std::string::npos) return src;
    std::string out = src;
    size_t at = 0;
    while ((at = out.find(key, at)) != std::string::npos) {
        size_t close = out.find(']', at);
        if (close == std::string::npos) break;
        if (out.compare(close + 1, 5, ".xywz") != 0) out.insert(close + 1, ".xywz");
        at = close;
    }
    return out;
}

}  // namespace gles
