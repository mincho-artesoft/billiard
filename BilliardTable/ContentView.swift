import SwiftUI
import MetalKit
import simd
import AVFoundation
import CoreHaptics

// MARK: - Sound Manager

class SoundManager {
    static let shared = SoundManager()
    var players: [String: AVAudioPlayer] = [:]

    func playSound(name: String) {
        if let url = Bundle.main.url(forResource: name, withExtension: "wav") {
            do {
                let player = try AVAudioPlayer(contentsOf: url)
                player.prepareToPlay()
                player.play()
                players[name] = player
            } catch {
                print("Failed to play \\(name): \\(error)")
            }
        } else {
            print("⚠️ Sound \\(name).wav not found in bundle.")
        }
    }
}

// MARK: - Metal Shader String

let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// -------------------------------------
//      0) Global Constants
// -------------------------------------
constant float PI = 3.1415926535;
constant float BALL_RADIUS = 0.47;
constant float BALL_DIAMETER = 2.0 * BALL_RADIUS; // = 0.94
constant float CUE_LENGTH = 2.5;
constant float FELT_HEIGHT = 0.01;

// K55 dimensions (1 unit = ~0.4178 inches)
constant float NOSE_HEIGHT    = 0.596; // 1.425 inches
constant float SUBRAIL_H      = 0.705; // 1.688 inches
constant float SUBRAIL_ANGLE  = 25.0 * (PI / 180.0); // 25°
constant float RAIL_BACK_DEPTH = 0.6;

// Table dimensions
constant float TABLE_HALF_WIDTH  = 7.6;
constant float TABLE_HALF_LENGTH = 13.6;

// Playing surface dimensions
constant float CUSHION_THICKNESS = 0.8356;
constant float PLAYING_HALF_WIDTH  = TABLE_HALF_WIDTH - CUSHION_THICKNESS;
constant float PLAYING_HALF_LENGTH = TABLE_HALF_LENGTH - CUSHION_THICKNESS;

// Pocket Radii
constant float CORNER_POCKET_R = 0.9;
constant float SIDE_POCKET_R   = 1.0;

// Rail positions
constant float RAIL_LENGTH_X = PLAYING_HALF_WIDTH;
constant float SIDE_RAIL_NEAR_Z_END = SIDE_POCKET_R;
constant float SIDE_RAIL_FAR_Z_END  = PLAYING_HALF_LENGTH;

// Pocket Center Locations
constant float2 CORNER_POCKET_CENTERS[4] = {
    float2(-PLAYING_HALF_WIDTH,  PLAYING_HALF_LENGTH), 
    float2( PLAYING_HALF_WIDTH,  PLAYING_HALF_LENGTH),
    float2(-PLAYING_HALF_WIDTH, -PLAYING_HALF_LENGTH), 
    float2( PLAYING_HALF_WIDTH, -PLAYING_HALF_LENGTH)
};

constant float2 SIDE_POCKET_CENTERS[2] = {
    float2(-PLAYING_HALF_WIDTH - RAIL_BACK_DEPTH, 0.0), 
    float2( PLAYING_HALF_WIDTH + RAIL_BACK_DEPTH, 0.0)
};

// -------------------------------------
//      1) Utility + Basic SDFs
// -------------------------------------
float hash(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i + float2(0.0, 0.0));
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(float2 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    float2 shift = float2(100.0);
    for (int i = 0; i < octaves; ++i) {
        v += a * noise(p);
        p = p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float3 hsvToRgb(float3 c) {
    float3 p = abs(fract(c.xxx + float3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(float3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

float prBoxDf(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCylinder(float3 p, float r, float h) {
    float2 d = abs(float2(length(p.xz), p.y)) - float2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float prRoundCylDf(float3 p, float r, float rt, float h) {
    float dxy = length(p.xy) - (r - (rt / 2.5) * p.z);
    float dz = abs(p.z) - h;
    return min(min(max(dxy + rt, dz), max(dxy, dz + rt)), length(float2(dxy, dz) + rt) - rt);
}

float3 rotateX(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
}

float3 rotateY(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

float3x3 qtToRMat(float4 q) {
    q = normalize(q);
    float s = q.w * q.w - 0.5;
    float3x3 m;
    m[0][0] = q.x * q.x + s; m[1][1] = q.y * q.y + s; m[2][2] = q.z * q.z + s;
    float a1 = q.x * q.y; float a2 = q.z * q.w; m[0][1] = a1 + a2; m[1][0] = a1 - a2;
    a1 = q.x * q.z; a2 = q.y * q.w; m[2][0] = a1 + a2; m[0][2] = a1 - a2;
    a1 = q.y * q.z; a2 = q.x * q.w; m[1][2] = a1 + a2; m[2][1] = a1 - a2;
    return 2.0 * m;
}

// -------------------------------------
//   2) Ball Intersection
// -------------------------------------
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ball {
    float2 position;
    float2 velocity;
    float4 quaternion;
    float height;
};

void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal,
             thread int &id, constant Ball* balls [[buffer(2)]]) {
    const int nBall = 16;
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    for (int n = 0; n < nBall; n++) {
        float ballCenterY = balls[n].height + BALL_RADIUS;
        float3 ballPos = float3(balls[n].position.x, ballCenterY, balls[n].position.y);
        if (isinf(balls[n].velocity.x)) continue;
        float3 u = ro - ballPos;
        float b = dot(rd, u);
        float w = b * b - dot(u, u) + BALL_RADIUS * BALL_RADIUS;
        if (w > 0.0) {
            float d = -b - sqrt(w);
            if (d > 0.0 && d < dist) {
                dist = d;
                normal = normalize(u + d * rd);
                id = n;
            }
        }
    }
}

// -------------------------------------
//   3) K55 Rail Profile & Pocket SDFs
// -------------------------------------
float sdK55Profile(float2 p, float scale) {
    float2 p0 = float2(0.0, NOSE_HEIGHT); // Nose point
    float2 p1 = float2(0.0, 0.0); // Bottom point (cushion face)
    float2 p2 = float2(RAIL_BACK_DEPTH * scale, SUBRAIL_H * scale); // Top-back point
    float2 p3 = float2(RAIL_BACK_DEPTH * scale, 0.0); // Bottom-back point

    float2 n_cushion = normalize(float2(0.43, -0.15));
    n_cushion.x = -n_cushion.x;
    float dist_cushion = dot(n_cushion, p - p0);

    float2 n_top = float2(0.0, 1.0);
    float dist_top = dot(n_top, p - float2(0.0, SUBRAIL_H * scale + 0.05));

    float2 n_subrail = float2(-sin(SUBRAIL_ANGLE), cos(SUBRAIL_ANGLE));
    float2 subrail_anchor = float2(0.15 * scale, NOSE_HEIGHT * scale - 0.2 * scale);
    float dist_subrail = dot(n_subrail, p - subrail_anchor);

    float2 n_back = float2(1.0, 0.0);
    float dist_back = dot(n_back, p - p3);

    float2 n_bottom = float2(0.0, -1.0);
    float dist_bottom = dot(n_bottom, p - p1);

    float dist = max(dist_cushion, dist_top);
    dist = max(dist, dist_back);
    dist = max(dist, dist_bottom);
    dist = max(dist, dist_subrail);

    return dist - 0.04;
}

float sdRailSegment(float3 p, float rail_start, float rail_end, float rail_axis_pos, int axis) {
    float3 p_relative = p - float3(0.0, FELT_HEIGHT, 0.0);
    float rail_coord, depth_coord_signed, height_coord;
    float half_length = abs(rail_end - rail_start) / 2.0;
    float mid_point = (rail_start + rail_end) / 2.0;
    height_coord = p_relative.y;

    float profile_dist;
    float length_dist;

    if (axis == 0) { // Head/Foot rails (along X)
        rail_coord = p_relative.x;
        depth_coord_signed = (p_relative.z - rail_axis_pos) * sign(rail_axis_pos);
        length_dist = abs(rail_coord - mid_point) - half_length;

        float taper_zone = 2.5;
        float taper_min_scale = 0.3;
        float dist_to_end = min(abs(rail_coord - rail_start), abs(rail_coord - rail_end));
        float taper = smoothstep(0.0, taper_zone, dist_to_end);
        float scale = mix(taper_min_scale, 1.0, taper);
        profile_dist = sdK55Profile(float2(depth_coord_signed, height_coord), scale);

        float fillet_radius = 0.15;
        profile_dist -= fillet_radius * (1.0 - taper);

        return max(profile_dist, length_dist);
    } else { // Side rails (along Z)
        rail_coord = p_relative.z;
        depth_coord_signed = (p_relative.x - rail_axis_pos) * sign(rail_axis_pos);
        length_dist = abs(rail_coord - mid_point) - half_length;

        float taper_zone = 2.5;
        float taper_min_scale = 0.3;
        float dist_to_end = min(abs(rail_coord - rail_start), abs(rail_coord - rail_end));
        float taper = smoothstep(0.0, taper_zone, dist_to_end);
        float scale = mix(taper_min_scale, 1.0, taper);
        profile_dist = sdK55Profile(float2(depth_coord_signed, height_coord), scale);

        float fillet_radius = 0.15;
        profile_dist -= fillet_radius * (1.0 - taper);

        return max(profile_dist, length_dist);
    }
}

float sdPocket(float3 p, float2 center, float radius) {
    float d = length(p.xz - center) - radius;
    return d;
}

// -------------------------------------
//   4) Scene Mapping & Normals
// -------------------------------------
float map(float3 p, thread float& hitType, thread float& pocketDist) {
    hitType = 0.0;
    pocketDist = 1000.0;

    float sceneRadius = max(TABLE_HALF_WIDTH, TABLE_HALF_LENGTH) + 2.0;
    float boundsDist = length(p.xz) - sceneRadius;
    boundsDist = max(boundsDist, abs(p.y) - 5.0);
    if (boundsDist > 1.0) {
        hitType = 0.0;
        return boundsDist;
    }

    float felt_thickness = 0.01;
    float felt_center_y = FELT_HEIGHT - felt_thickness;
    float dFelt = prBoxDf(p - float3(0.0, felt_center_y, 0.0), 
                         float3(PLAYING_HALF_WIDTH, felt_thickness, PLAYING_HALF_LENGTH));

    float dHeadRail = sdRailSegment(p, -RAIL_LENGTH_X, RAIL_LENGTH_X, PLAYING_HALF_LENGTH, 0);
    float dFootRail = sdRailSegment(p, -RAIL_LENGTH_X, RAIL_LENGTH_X, -PLAYING_HALF_LENGTH, 0);
    float dLeftRailFar = sdRailSegment(p, SIDE_RAIL_NEAR_Z_END, SIDE_RAIL_FAR_Z_END - CORNER_POCKET_R, 
                                      -PLAYING_HALF_WIDTH, 1);
    float dLeftRailNear = sdRailSegment(p, -SIDE_RAIL_FAR_Z_END + CORNER_POCKET_R, -SIDE_RAIL_NEAR_Z_END, 
                                       -PLAYING_HALF_WIDTH, 1);
    float dLeftRail = min(dLeftRailFar, dLeftRailNear);
    float dRightRailFar = sdRailSegment(p, SIDE_RAIL_NEAR_Z_END, SIDE_RAIL_FAR_Z_END - CORNER_POCKET_R, 
                                       PLAYING_HALF_WIDTH, 1);
    float dRightRailNear = sdRailSegment(p, -SIDE_RAIL_FAR_Z_END + CORNER_POCKET_R, -SIDE_RAIL_NEAR_Z_END, 
                                        PLAYING_HALF_WIDTH, 1);
    float dRightRail = min(dRightRailFar, dRightRailNear);

    float dRails = min(min(dHeadRail, dFootRail), min(dLeftRail, dRightRail));

    float dPocketCorners = 1000.0;
    for (int i = 0; i < 4; ++i) {
        dPocketCorners = min(dPocketCorners, sdPocket(p, CORNER_POCKET_CENTERS[i], CORNER_POCKET_R));
    }
    float dPocketSides = 1000.0;
    for (int i = 0; i < 2; ++i) {
        dPocketSides = min(dPocketSides, sdPocket(p, SIDE_POCKET_CENTERS[i], SIDE_POCKET_R));
    }
    float dPocketsXZ = min(dPocketCorners, dPocketSides);
    pocketDist = dPocketsXZ;

    float dSolid = min(dFelt, dRails);

    if (dPocketsXZ < 0.01 && p.y < FELT_HEIGHT) {
        hitType = 3.0;
        return dSolid;
    }

    float d = max(dSolid, -dPocketsXZ);

    if (dSolid == dFelt) {
        hitType = 1.0;
    } else if (dRails < dFelt) {
        hitType = 2.0;
    }

    return d;
}

float3 getNormal(float3 p, thread float& hitType, thread float& pocketDist) {
    float2 e = float2(0.0005, 0.0);
    float ht_ignore, pd_ignore;
    if (hitType == 3.0) {
        return float3(0.0, -1.0, 0.0);
    }
    return normalize(float3(
        map(p + e.xyy, ht_ignore, pd_ignore) - map(p - e.xyy, ht_ignore, pd_ignore),
        map(p + e.yxy, ht_ignore, pd_ignore) - map(p - e.yxy, ht_ignore, pd_ignore),
        map(p + e.yyx, ht_ignore, pd_ignore) - map(p - e.yyx, ht_ignore, pd_ignore)
    ));
}

// -------------------------------------
//   5) showScene
// -------------------------------------
float3 showScene(float3 ro, float3 rd,
                 float time,
                 float cueOffset,
                 float2 cueTipOffset,
                 constant Ball* balls,
                 int cueVisible,
                 float cueAngle,
                 float2 cue3DRotate,
                 float strikeAnimationOffset,
                 float3 cueBallWorldPos)
{
    float3 col = float3(0.05, 0.05, 0.1);
    float t = 0.0;
    const float maxDist = 50.0;

    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    float hitDist = maxDist;
    float hitType = 0.0;
    float pocketDistAtHit = 1000.0;
    float3 cueHitPos;

    for (int i = 0; i < 80; i++) { // Reduced iterations for better performance
        float3 p = ro + rd * t;
        float dCueStick = maxDist;
        if (cueVisible != 0 && !isinf(balls[0].velocity.x)) {
            float3 pc = p - cueBallWorldPos;
            float baseAngle = sin(time * 0.5) * 0.1;
            pc = rotateX(pc, cue3DRotate.y);
            float finalYaw = baseAngle + cue3DRotate.x;
            pc = rotateY(pc, finalYaw);
            float tipOffset = CUE_LENGTH;
            float maxCueOffset = -BALL_RADIUS;
            float strikeZ = (cueOffset - strikeAnimationOffset);
            strikeZ = max(strikeZ, 0.0); // prevents overshooting the white ball
            pc.z += (maxCueOffset - strikeZ - tipOffset);            pc = rotateY(pc, PI);
            pc.x -= cueTipOffset.x;
            pc.y -= cueTipOffset.y;
            dCueStick = prRoundCylDf(pc, 0.1 - (0.015 / 2.5) * (pc.z + tipOffset), 0.05, CUE_LENGTH);
            cueHitPos = pc;
        }

        float mapHitType = 0.0;
        float mapPocketDist = 1000.0;
        float dEnv = map(p, mapHitType, mapPocketDist);

        float d = min(dEnv, dCueStick);
        if (d < 0.0005 || t > dstBall) {
            if (dstBall <= t + 0.001 && dstBall < maxDist) {
                hitDist = dstBall;
                hitType = 4.0;
            } else if (dEnv <= dCueStick) {
                hitDist = t;
                hitType = mapHitType;
                pocketDistAtHit = mapPocketDist;
            } else if (dCueStick < maxDist) {
                hitDist = t;
                hitType = 5.0;
            } else {
                hitDist = maxDist;
                hitType = 0.0;
            }
            break;
        }
        t += max(d * 0.7, 0.001); // Adaptive step size
        if (t > maxDist) break;
    }

    if (hitType > 0.0) {
        float3 p = ro + rd * hitDist;
        float3 n;
        float3 lightPos = float3(0.0, 20.0, 0.0);
        float3 lightDir = normalize(lightPos - p);
        float ambient = 0.4;

        if (hitType == 1.0) { // Felt
            float ignoredHitType;
            n = getNormal(p, ignoredHitType, pocketDistAtHit);
            float2 feltUV = p.xz * 0.5;
            float feltNoise = fbm(feltUV, 4);
            float fiberDetail = noise(feltUV * 15.0);
            float3 feltBaseColor = float3(0.1, 0.5, 0.2);
            float3 fiberColor = float3(0.05, 0.3, 0.1);
            float fiberMix = smoothstep(0.6, 0.8, fiberDetail);
            float3 feltColor = mix(feltBaseColor, fiberColor, fiberMix);
            feltColor *= (0.8 + 0.2 * feltNoise);
            float diff = max(dot(n, lightDir), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, lightDir), 0.0), 8.0);
            col = feltColor * (ambient + (1.0 - ambient) * diff);
            col += float3(0.05) * spec * (0.5 + 0.5 * feltNoise);
        } else if (hitType == 2.0) { // Rail
            float ignoredHitType;
            n = getNormal(p, ignoredHitType, pocketDistAtHit);

            float2 woodUV = (p.x > p.z) ? p.xy : p.zy;
            float woodGrain = fbm(woodUV * 5.0, 4);
            float grainDetail = noise(woodUV * 20.0);
            float3 woodBaseColor = float3(0.45, 0.3, 0.2);
            float3 woodGrainColor = float3(0.3, 0.15, 0.1);
            float3 railColor = mix(woodBaseColor, woodGrainColor, smoothstep(0.3, 0.7, woodGrain));

            float3 normalPerturb = float3(noise(woodUV * 10.0) - 0.5, noise(woodUV * 10.0 + 100.0) - 0.5, 0.0) * 0.1;
            n = normalize(n + normalPerturb);

            float diff = max(dot(n, lightDir), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, lightDir), 0.0), 64.0);
            col = railColor * (ambient + (1.0 - ambient) * diff);
            col += float3(0.5) * spec * (0.7 + 0.3 * grainDetail);
        } else if (hitType == 3.0) { // Pocket
            col = float3(0.01, 0.01, 0.01);
        } else if (hitType == 4.0) { // Ball
            n = ballNormal;
            int id = ballId;
            if (id == 0) { col = float3(1.0); }
            else {
                bool isStriped = (id >= 9);
                float3 baseColor;
                if (id == 8) { baseColor = float3(0.0); }
                else {
                    float hue = 0.0;
                    // Adjusted hues for standard billiard ball colors
                    if (id == 1 || id == 9) hue = 60.0 / 360.0;  // Yellow
                    if (id == 2 || id == 10) hue = 240.0 / 360.0; // Blue
                    if (id == 3 || id == 11) hue = 0.0 / 360.0;   // Red
                    if (id == 4 || id == 12) hue = 300.0 / 360.0; // Purple
                    if (id == 5 || id == 13) hue = 30.0 / 360.0;  // Orange
                    if (id == 6 || id == 14) hue = 120.0 / 360.0; // Green
                    if (id == 7 || id == 15) hue = 330.0 / 360.0; // Maroon
                    baseColor = hsvToRgb(float3(hue, 0.9, 1.0));
                }
                float3x3 rotMat = qtToRMat(balls[id].quaternion);
                float3 rotatedNormal = rotMat * n;
                float2 uv = float2(atan2(rotatedNormal.x, rotatedNormal.z) / (2.0 * PI) + 0.5, 
                                 acos(rotatedNormal.y) / PI);
                if (isStriped && id != 8) {
                    float stripeWidth = 0.3;
                    float stripePattern = sin(uv.x * 10.0) * 0.5 + 0.5;
                    col = mix(float3(1.0), baseColor, step(stripeWidth, uv.y) * step(uv.y, 1.0 - stripeWidth));
                } else {
                    col = baseColor;
                }
                float2 circleCenter = float2(0.5, 0.5);
                float circleRadius = 0.2;
                float distToCenter = length(uv - circleCenter);
                if (distToCenter < circleRadius && id != 0) {
                    col = float3(1.0);
                }
            }
            float diff = max(dot(n, lightDir), 0.0);
            col *= (ambient + (1.0 - ambient) * diff);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, lightDir), 0.0), 24.0);
            col += float3(0.3) * spec;
        } else if (hitType == 5.0) { // Cue Stick
            float3 eps = float3(0.0005, 0.0, 0.0);
            n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, CUE_LENGTH) - 
                prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, CUE_LENGTH),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, CUE_LENGTH) - 
                prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, CUE_LENGTH),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, CUE_LENGTH) - 
                prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, CUE_LENGTH)
            ));
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);
            float diff = max(dot(n, lightDir), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, lightDir), 0.0), 16.0);
            col *= (0.3 + 0.7 * diff);
            col += float3(0.2) * spec;
        }
    }

    return clamp(col, 0.0, 1.0);
}

// -------------------------------------
//   6) Vertex & Fragment Shaders
// -------------------------------------
vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), 
                                     float2(-1.0, 1.0), float2(1.0, 1.0) };
    constexpr float2 uvs[4] = { float2(0.0, 0.0), float2(1.0, 0.0), 
                               float2(0.0, 1.0), float2(1.0, 1.0) };
    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant float2 &resolution [[buffer(0)]],
                               constant float &time [[buffer(1)]],
                               constant Ball* balls [[buffer(2)]],
                               constant float &cueOffset [[buffer(3)]],
                               constant int &cueVisible [[buffer(4)]],
                               constant float2 &cueTipOffset [[buffer(5)]],
                               constant float &cueAngle [[buffer(6)]],
                               constant float2 &cue3DRotate [[buffer(7)]],
                               constant float &strikeAnimationOffset [[buffer(8)]],
                               constant float3 &cueBallWorldPos [[buffer(9)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;
    float angle = time * 0.1;
    float3 camPos = float3(sin(angle) * 25.0, 12.0, cos(angle) * 25.0);
    float3 camTarget = float3(0.0, FELT_HEIGHT, 0.0);
    float3 ww = normalize(camTarget - camPos);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.7;
    float3 rd = normalize(ww + uu * uv.x * fov + vv * uv.y * fov);
    float3 col = showScene(camPos, rd, time, cueOffset, cueTipOffset, balls,
                           cueVisible, cueAngle, cue3DRotate, strikeAnimationOffset, cueBallWorldPos);
    return float4(col, 1.0);
}

fragment float4 behindBallFragmentShader(VertexOut in [[stage_in]],
                                         constant float2 &resolution   [[buffer(0)]],
                                         constant float &time          [[buffer(1)]],
                                         constant float3 &cameraPos    [[buffer(2)]],
                                         constant float3 &cameraTarget [[buffer(3)]],
                                         constant Ball*  balls         [[buffer(4)]],
                                         constant float  &cueOffset    [[buffer(5)]],
                                         constant int    &cueVisible   [[buffer(6)]],
                                         constant float2 &cueTipOffset [[buffer(7)]],
                                         constant float  &cueAngle     [[buffer(8)]],
                                         constant float2 &cue3DRotate  [[buffer(9)]],
                               constant float  &strikeAnimationOffset [[buffer(10)]],
                               constant float3 &cueBallWorldPos [[buffer(11)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;
    float3 ro = cameraPos;
    float3 target = cameraTarget;
    float3 ww = normalize(target - ro);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.8;
    float3 rd = normalize(ww + uu * uv.x * fov + vv * uv.y * fov);
    float3 col = showScene(ro, rd, time, cueOffset, cueTipOffset, balls, 
                          cueVisible, cueAngle, cue3DRotate, strikeAnimationOffset, cueBallWorldPos);
    return float4(col, 1.0);
}

fragment float4 thirdBallFragmentShader(VertexOut in [[stage_in]],
                                        constant float2 &resolution   [[buffer(0)]],
                                        constant float &time          [[buffer(1)]],
                                        constant float3 &cameraPos    [[buffer(2)]],
                                        constant float3 &cameraTarget [[buffer(3)]],
                                        constant Ball*  balls         [[buffer(4)]],
                                        constant float  &cueOffset    [[buffer(5)]],
                                        constant int    &cueVisible   [[buffer(6)]],
                                        constant float2 &cueTipOffset [[buffer(7)]],
                                        constant float  &cueAngle     [[buffer(8)]],
                                        constant float2 &cue3DRotate  [[buffer(9)]],
                                        constant float  &strikeAnimationOffset [[buffer(10)]],
                                        constant float3 &cueBallWorldPos [[buffer(11)]]) {
    float2 uv = 2.0 * in.uv - 1.0;
    uv.x *= resolution.x / resolution.y;
    float3 ro = cameraPos;
    float3 target = cameraTarget;
    float3 ww = normalize(target - ro);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.8;
    float3 rd = normalize(ww + uu * uv.x * fov + vv * uv.y * fov);
    float3 col = showScene(ro, rd, time, cueOffset, cueTipOffset, balls, 
                          cueVisible, cueAngle, cue3DRotate, strikeAnimationOffset, cueBallWorldPos);
    return float4(col, 1.0);
}
"""

// MARK: - Ball Types

struct BallData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var height: Float
    var verticalVelocity: Float
    var angularVelocity: SIMD3<Float>
    var quaternion: SIMD4<Float>
}

// MARK: - Utility Functions

func quaternion(from axis: SIMD3<Float>, angle: Float) -> SIMD4<Float> {
    let half = angle * 0.5
    let s = sin(half)
    return SIMD4<Float>(axis.x * s, axis.y * s, axis.z * s, cos(half))
}

func quaternionMultiply(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> SIMD4<Float> {
    SIMD4<Float>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    )
}

func clamp<T: Comparable>(_ value: T, lower: T, upper: T) -> T {
    return max(lower, min(value, upper))
}

// MARK: - Physics Simulator

final class PhysicsEngine {
    var balls: [BallData] = []

    // Table dimensions
    let ballRadius: Float = 0.47
    let cushionEdgeX: Float = 7.6 - 0.8356
    let cushionEdgeZ: Float = 13.6 - 0.8356
    
    enum GameMode { case eightBall, nineBall }
    var gameMode: GameMode = .eightBall

    private var hitSoundPlayed = false
    
    init() {
        resetRack()
    }

    func resetRack() {
        let r = ballRadius
        let d = r * 2
        let rowSpacing = d * sqrt(3) / 2
        let headZ: Float = 13.6 / 2
        let footZ: Float = -13.6 / 2
        let identity = SIMD4<Float>(0, 0, 0, 1)
        
        balls = []
        balls.removeAll()
        // Cue ball
        balls.append(BallData(position: [0, headZ], velocity: .zero, height: 0.01,
                              verticalVelocity: 0, angularVelocity: .zero, quaternion: identity))
        
        let startZ = footZ + rowSpacing * 2
        
        if gameMode == .eightBall {
            for row in 0..<5 {
                for i in 0...row {
                    let x = (Float(i) - Float(row)/2) * d
                    let z = startZ + Float(row) * rowSpacing
                    balls.append(BallData(position: [x, z], velocity: .zero, height: 0.01,
                                          verticalVelocity: 0, angularVelocity: .zero, quaternion: identity))
                }
            }
        } else if gameMode == .nineBall {
            var count = 1
            for row in 0..<5 {
                for i in 0...row where count <= 9 {
                    let x = (Float(i) - Float(row)/2) * d
                    let z = startZ + Float(row) * rowSpacing
                    balls.append(BallData(position: [x, z], velocity: .zero, height: 0.01,
                                          verticalVelocity: 0, angularVelocity: .zero, quaternion: identity))
                    count += 1
                }
            }
        }
        while balls.count < 16 {
            balls.append(BallData(position: [0, 0],
                                  velocity: [Float.infinity, Float.infinity],
                                  height: 0.01,
                                  verticalVelocity: 0,
                                  angularVelocity: .zero,
                                  quaternion: SIMD4<Float>(0, 0, 0, 1)))
        }
    }

    func update(delta: Float) {
        let g: Float = 9.81
        let maxVelocity = balls.map { simd_length($0.velocity) }.max() ?? 0
        let substeps = Int(clamp(6 + maxVelocity * 4, lower: 6, upper: 20))
        let dt = delta / Float(substeps)

        for _ in 0..<substeps {
            for i in balls.indices {
                var b = balls[i]
                if b.velocity.x.isInfinite { continue }
                
                // Gravity + vertical motion
                b.verticalVelocity -= g * dt
                b.height += b.verticalVelocity * dt
                if b.height <= 0.01 {
                    b.height = 0.01
                    b.verticalVelocity = -b.verticalVelocity * 0.5
                    if abs(b.verticalVelocity) < 0.05 {
                        b.verticalVelocity = 0
                    }
                }
                
                // Friction (very simple)
                b.velocity *= 0.995
                b.angularVelocity *= 0.995
                
                b.position += b.velocity * dt
                
                // Spin
                let wMag = simd_length(b.angularVelocity)
                if wMag > 0 {
                    let axis = b.angularVelocity / wMag
                    let angle = wMag * dt
                    let dq = quaternion(from: axis, angle: angle)
                    b.quaternion = quaternionMultiply(dq, b.quaternion)
                }
                
                // Cushion bounce
                if abs(b.position.x) > cushionEdgeX - ballRadius {
                    b.position.x = min(max(b.position.x, -cushionEdgeX + ballRadius), cushionEdgeX - ballRadius)
                    b.velocity.x *= -0.9
                }
                
                if abs(b.position.y) > cushionEdgeZ - ballRadius {
                    b.position.y = min(max(b.position.y, -cushionEdgeZ + ballRadius), cushionEdgeZ - ballRadius)
                    b.velocity.y *= -0.9
                }
                
                // Pocket detection logic
                if checkPocket(pos: b.position, height: b.height) {
                    SoundManager.shared.playSound(name: "FourBallPoint")
                    b.velocity = [Float.infinity, Float.infinity]
                    b.verticalVelocity = 0.0
                    b.angularVelocity = .zero
                    b.position = .zero
                    b.height = 0.01
                }
                if i == 0, simd_length(b.velocity) > 0.1, !hitSoundPlayed {
                    SoundManager.shared.playSound(name: "Hit01")
                    hitSoundPlayed = true
                }
                
                balls[i] = b
            }

            // Ball-ball collisions (basic)
            for i in 0..<balls.count {
                for j in i+1..<balls.count {
                    var bi = balls[i]
                    var bj = balls[j]
                    if bi.velocity.x.isInfinite || bj.velocity.x.isInfinite { continue }

                    let delta = bj.position - bi.position
                    let dist = simd_length(delta)
                    let minDistance = 2 * ballRadius
                    let epsilon: Float = 0.0001
                    if dist < minDistance - epsilon {
                        let normal = delta / dist
                        let relative = bj.velocity - bi.velocity
                        let separating = simd_dot(relative, normal)

                        if separating < 0 {
                            let impulse = -1.5 * separating / 2
                            bi.velocity -= impulse * normal
                            bj.velocity += impulse * normal

                            // Separate
                            let overlap = 2 * ballRadius - dist
                            let correction = 0.6 * overlap * normal
                            bi.position -= correction
                            bj.position += correction
                        }

                        balls[i] = bi
                        balls[j] = bj
                    }
                }
            }
        }
        let moving = balls.contains { simd_length($0.velocity) > 0.05 }
        if !moving {
            hitSoundPlayed = false
        }
    }
    
    func checkPocket(pos: SIMD2<Float>, height: Float) -> Bool {
        let railBackDepth: Float = 0.6
        let pocketPositions: [SIMD2<Float>] = [
            SIMD2<Float>(-cushionEdgeX,  cushionEdgeZ),
            SIMD2<Float>( cushionEdgeX,  cushionEdgeZ),
            SIMD2<Float>(-cushionEdgeX - railBackDepth,  0.0),
            SIMD2<Float>( cushionEdgeX + railBackDepth,  0.0),
            SIMD2<Float>(-cushionEdgeX, -cushionEdgeZ),
            SIMD2<Float>( cushionEdgeX, -cushionEdgeZ)
        ]
        
        let cornerPocketRadius: Float = 0.9
        let sidePocketRadius: Float = 1.0
        
        for (index, pocketPos) in pocketPositions.enumerated() {
            let radius = (index == 2 || index == 3) ? sidePocketRadius : cornerPocketRadius
            if simd_length(pos - pocketPos) < radius && height <= 0.01 + ballRadius {
                return true
            }
        }
        return false
    }
}

// MARK: - Cue Control & Input

final class CueController {
    var aimAngle: Float = 0        // Horizontal aim (in radians)
    var verticalAngle: Float = 0   // For 3D pitch if needed
    var tipOffset: SIMD2<Float> = .zero
    var cuePull: Float = 0         // How far the cue is pulled back
    var maxPull: Float = 2.0
    var isCharging = false
    var isShotFired = false
    var baseSpeed: Float = 50.0  // Default base speed for cue shots
    var strikeStartPosition: SIMD2<Float> = .zero
    
    var strikeTimer: Float = 0.0
    let strikeDuration: Float = 0.2

    var strikeAnimating: Bool {
        isShotFired && strikeTimer < strikeDuration
    }
    
    var strikeAnimationOffset: Float {
        if !strikeAnimating { return 0 }
        let t = strikeTimer / strikeDuration
        return maxPull * (1 - t)  // smoothly animates from cuePull back to 0
    }

    func update(delta: Float) {
        if isCharging {
            cuePull += delta * 1.5
            cuePull = min(cuePull, maxPull)
        }

        if isShotFired {
            strikeTimer += delta
            if strikeTimer > strikeDuration {
                isShotFired = false
                strikeTimer = 0.0
            }
        }
    }

    func fireStrike(onto whiteBall: inout BallData) {
        let baseSpeed = self.baseSpeed
        let powerFactor = cuePull / maxPull
        let power = baseSpeed * pow(powerFactor, 0.75)  // nonlinear, punchy

        var dir = SIMD2<Float>(sin(aimAngle), -cos(aimAngle))
        let spinAxis = SIMD3<Float>(-tipOffset.y, 0, tipOffset.x)
        let spin = spinAxis * (power * 2.5)
        strikeStartPosition = whiteBall.position
        whiteBall.velocity = dir * power
        whiteBall.angularVelocity = spin
        whiteBall.verticalVelocity = max(0, -sin(verticalAngle) * power * 0.5)

        cuePull = 0
        isCharging = false
        isShotFired = true
    }

    func resetShot() {
        isShotFired = false
        cuePull = 0
    }
}

// MARK: - Renderer + Simulation Glue

final class BilliardSimulation: ObservableObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var library: MTLLibrary!
    private var pipeline: MTLRenderPipelineState!

    private var drawableSize: CGSize = .zero
    private var time: Float = 0
    
    var showCueValue: Int32 = 1
    let physics = PhysicsEngine()
    let cue = CueController()

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal not supported")
        }

        self.device = device
        self.commandQueue = queue

        // Compile shader
        do {
            library = try device.makeLibrary(source: metalShaderSource, options: nil)
            let vertexFunc = library.makeFunction(name: "vertexShader")
            let fragFunc = library.makeFunction(name: "fragmentShader")

            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vertexFunc
            desc.fragmentFunction = fragFunc
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm

            pipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Shader compile error: \(error)")
        }
    }

    func update(drawableSize: CGSize, topDown: Bool) {
        self.drawableSize = drawableSize
        let delta: Float = 1.0 / 60.0
        time += delta
 
        cue.update(delta: delta)
        physics.update(delta: delta)
 
        if cue.isCharging || cue.strikeAnimating {
            showCueValue = 1
        } else {
            let anyMoving = physics.balls.contains { simd_length($0.velocity) > 0.05 }
            showCueValue = anyMoving ? 0 : 1
        }
    }

    func render(to view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        encoder.setRenderPipelineState(pipeline)

        // Required uniform buffers
        var resolution = SIMD2<Float>(Float(drawableSize.width), Float(drawableSize.height))
        var currentTime = time
        var cueOffset = cue.cuePull
        var strikeAnimationOffset = cue.strikeAnimationOffset
        var cueVisible = showCueValue
        var cueTipOffset = cue.tipOffset
        var cueAngle = cue.aimAngle
        var cue3DRotate = SIMD2<Float>(cue.aimAngle, cue.verticalAngle)
        let strikePos = cue.strikeAnimating ? cue.strikeStartPosition : physics.balls[0].position
        var cueBallWorldPos = SIMD3<Float>(strikePos.x, physics.balls[0].height + 0.47, strikePos.y)

        // Allocate Metal buffers
        let ballsBuffer = device.makeBuffer(bytes: physics.balls,
                                            length: MemoryLayout<BallData>.stride * physics.balls.count,
                                            options: .storageModeShared)!

        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        encoder.setFragmentBytes(&currentTime, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setFragmentBuffer(ballsBuffer, offset: 0, index: 2)
        encoder.setFragmentBytes(&cueOffset, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setFragmentBytes(&cueVisible, length: MemoryLayout<Int32>.stride, index: 4)
        encoder.setFragmentBytes(&cueTipOffset, length: MemoryLayout<SIMD2<Float>>.stride, index: 5)
        encoder.setFragmentBytes(&cueAngle, length: MemoryLayout<Float>.stride, index: 6)
        encoder.setFragmentBytes(&cue3DRotate, length: MemoryLayout<SIMD2<Float>>.stride, index: 7)
        encoder.setFragmentBytes(&strikeAnimationOffset, length: MemoryLayout<Float>.stride, index: 8)
        encoder.setFragmentBytes(&cueBallWorldPos, length: MemoryLayout<SIMD3<Float>>.stride, index: 9)
        
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Gesture Input

    func handleDrag(value: DragGesture.Value) {
        let delta = value.translation
        let dx = Float(delta.width)
        let targetAngle = cue.aimAngle - dx * 0.0025
        cue.aimAngle = simd_mix(cue.aimAngle, targetAngle, 0.1)
        cue.isCharging = true
    }

    func handleRelease() {
        if !cue.isShotFired {
            if var white = physics.balls.first {
                cue.fireStrike(onto: &white)
                physics.balls[0] = white
            }
        }
    }

    func start() {
        // Called on .onAppear
        time = 0
    }
}

// MARK: - Main SwiftUI View

struct ContentView: View {
    @StateObject private var sim = BilliardSimulation()
    @State private var isTopDown = false

    var body: some View {
        ZStack {
            BilliardMetalView(simulation: sim, isTopDown: isTopDown)
                .edgesIgnoringSafeArea(.all)

            VStack {
            HStack {
                    Button(action: {
                        isTopDown.toggle()
                    }) {
                        Text(isTopDown ? "🔄 Orbit Camera" : "🔭 Top-Down View")
                            .padding()
                            .background(Color.black.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                    Spacer()
                    Button(action: {
                        if sim.physics.gameMode == .eightBall {
                            sim.physics.gameMode = .nineBall
                        } else {
                            sim.physics.gameMode = .eightBall
                        }
                        sim.physics.resetRack()
                        sim.cue.resetShot()
                    }) {
                        Text(sim.physics.gameMode == .eightBall ? "🎱 9-Ball Mode" : "🎱 8-Ball Mode")
                            .padding()
                            .background(Color.black.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                    Button(action: {
                        SoundManager.shared.playSound(name: "ButtonDown")
                        sim.physics.resetRack()
                        sim.cue.resetShot()
                    }) {
                        Text("🔁 Rerack")
                            .padding()
                            .background(Color.black.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
            }
                .padding()
                Spacer()
            }
        }
        .onAppear {
            sim.start()
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in sim.handleDrag(value: value) }
                .onEnded { _ in sim.handleRelease() }
        )
    }
}

// MARK: - Metal View Integration

struct BilliardMetalView: UIViewRepresentable {
    let simulation: BilliardSimulation
    let isTopDown: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(simulation: simulation, isTopDown: isTopDown)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = simulation.device
        view.delegate = context.coordinator
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.1, 1)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.isTopDown = isTopDown
    }

    class Coordinator: NSObject, MTKViewDelegate {
        var simulation: BilliardSimulation
        var isTopDown: Bool

        init(simulation: BilliardSimulation, isTopDown: Bool) {
            self.simulation = simulation
            self.isTopDown = isTopDown
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            simulation.update(drawableSize: view.drawableSize, topDown: isTopDown)
            simulation.render(to: view)
        }
    }
}
