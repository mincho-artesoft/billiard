import SwiftUI
import MetalKit
import simd

// MARK: - Metal Shader Source
let metalShader = """
#include <metal_stdlib>
using namespace metal;

// -------------------------------------
//      0) Global Constants
// -------------------------------------
constant float PI = 3.1415926535;
constant float BALL_RADIUS = 0.47;
constant float BALL_DIAMETER = 2.0 * BALL_RADIUS; // = 0.94
constant float CUE_LENGTH = 2.5; // Standard cue length in shader units

// Define K55 dimensions in shader units (1 inch = 0.4178 units approx)
constant float NOSE_HEIGHT    = 0.609; // 1.458 inches
constant float SUBRAIL_H      = 0.705; // 1.688 inches (Absolute height, not currently used explicitly)
constant float SUBRAIL_ANGLE  = 23.5 * (PI / 180.0); // In radians
constant float RAIL_BACK_DEPTH = 0.6; // Estimated depth of cushion back

// Table dimensions
constant float TABLE_HALF_WIDTH  = 7.6;
constant float TABLE_HALF_LENGTH = 13.6;

// Pocket Radii (scaled from WPA specs)
constant float CORNER_POCKET_R = 0.96;
constant float SIDE_POCKET_R   = 1.06;

// Pocket Center Locations
constant float2 CORNER_POCKET_CENTERS[4] = {
    float2(-TABLE_HALF_WIDTH,  TABLE_HALF_LENGTH), float2( TABLE_HALF_WIDTH,  TABLE_HALF_LENGTH),
    float2(-TABLE_HALF_WIDTH, -TABLE_HALF_LENGTH), float2( TABLE_HALF_WIDTH, -TABLE_HALF_LENGTH)
};
constant float2 SIDE_POCKET_CENTERS[2] = {
    float2(-TABLE_HALF_WIDTH, 0.0), float2( TABLE_HALF_WIDTH, 0.0)
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

// Box SDF - centered at origin
float prBoxDf(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// Cylinder SDF - centered at origin, aligned with Y axis
float sdCylinder(float3 p, float r, float h) {
    float2 d = abs(float2(length(p.xz),p.y)) - float2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// Restored original rounded cylinder SDF for the cue stick
float prRoundCylDf(float3 p, float r, float rt, float h) {
    float dxy = length(p.xy) - (r - (rt / 2.5) * p.z);
    float dz = abs(p.z) - h;
    return min(min(max(dxy + rt, dz), max(dxy, dz + rt)), length(float2(dxy, dz) + rt) - rt);
}

float3 rotateX(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(v.x, c*v.y - s*v.z, s*v.y + c*v.z);
}

float3 rotateY(float3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
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
//   2) Ball Intersection (Corrected Height)
// -------------------------------------
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ball {
    float2 position;
    float2 velocity;
    float4 quaternion;
    float height; // Height of the *bottom* of the ball from physics
};

void ballHit(float3 ro, float3 rd, thread float &dist, thread float3 &normal,
             thread int &id, constant Ball* balls [[buffer(2)]]) {
    const int nBall = 16;
    dist = 50.0;
    normal = float3(0.0);
    id = -1;
    for (int n = 0; n < nBall; n++) {
        // Corrected ball center Y coordinate
        float ballCenterY = balls[n].height + BALL_RADIUS - 0.6; // Adjust for felt height
        float3 ballPos = float3(balls[n].position.x, ballCenterY, balls[n].position.y);

        // Check for pocketed balls (marked with infinite velocity in Swift)
        if (isinf(balls[n].velocity.x)) continue;

        float3 u = ro - ballPos;
        float b = dot(rd, u);
        float w = b*b - dot(u, u) + BALL_RADIUS*BALL_RADIUS;
        if (w > 0.0) {
            float d = -b - sqrt(w);
            if (d > 0.0 && d < dist) {
                dist = d;
                normal = normalize(u + d*rd);
                id = n;
            }
        }
    }
}

// -------------------------------------
//   3) K55 Rail Profile & Pocket SDFs
// -------------------------------------
float sdK55Profile(float2 p) {
    // Plane 1: Cushion Face
    float2 n_cushion = normalize(float2(0.4508, -0.133));
    n_cushion.x = -n_cushion.x;
    float dist_cushion = dot(n_cushion, p - float2(0.0, NOSE_HEIGHT));

    // Plane 2: Top Face
    float top_y = NOSE_HEIGHT + 0.05;
    float2 n_top = float2(0.0, 1.0);
    float dist_top = dot(n_top, p - float2(0.0, top_y));

    // Plane 3: Subrail Face
    float2 n_subrail = float2(-sin(SUBRAIL_ANGLE), cos(SUBRAIL_ANGLE));
    float2 subrail_anchor = float2(0.1, NOSE_HEIGHT - 0.15);
    float dist_subrail = dot(n_subrail, p - subrail_anchor);

    // Plane 4: Back Face
    float2 n_back = float2(1.0, 0.0);
    float dist_back = dot(n_back, p - float2(RAIL_BACK_DEPTH, 0.0));

    // Plane 5: Bottom Face (Table bed)
    float2 n_bottom = float2(0.0, -1.0);
    float dist_bottom = dot(n_bottom, p - float2(0.0, 0.0));

    float dist = max(dist_cushion, dist_top);
    dist = max(dist, dist_back);
    dist = max(dist, dist_bottom);
    dist = max(dist, dist_subrail);

    return dist;
}

float sdRailSegmentCorrected(float3 p, float rail_start, float rail_end, float rail_axis_pos, int axis) {
    float rail_coord, depth_coord_signed, height_coord;
    float half_length = abs(rail_end - rail_start) / 2.0;
    float mid_point = (rail_start + rail_end) / 2.0;

    height_coord = p.y;

    if (axis == 0) {
        rail_coord = p.x;
        depth_coord_signed = (p.z - rail_axis_pos) * sign(rail_axis_pos);
        float length_dist = abs(rail_coord - mid_point) - half_length;
        float profile_dist = sdK55Profile(float2(depth_coord_signed, height_coord));
        return max(profile_dist, length_dist);
    } else {
        rail_coord = p.z;
        depth_coord_signed = (p.x - rail_axis_pos) * sign(rail_axis_pos);
        float length_dist = abs(rail_coord - mid_point) - half_length;
        float profile_dist = sdK55Profile(float2(depth_coord_signed, height_coord));
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
    hitType = 0.0; // 0=miss, 1=felt, 2=rail, 3=pocket_hole
    pocketDist = 1000.0;

    // Bounding Sphere (Keep as is)
    float sceneRadius = max(TABLE_HALF_WIDTH, TABLE_HALF_LENGTH) + 2.0;
    float boundsDist = length(p.xz) - sceneRadius;
    boundsDist = max(boundsDist, abs(p.y) - 5.0);
    if (boundsDist > 1.0) {
        hitType = 0.0;
        return boundsDist;
    }

    // Felt SDF (Adjusted to y = -0.6)
    float dFelt = prBoxDf(p - float3(0.0, -0.6, 0.0), float3(TABLE_HALF_WIDTH, 0.01, TABLE_HALF_LENGTH));

    // Rail Segment SDFs (Adjust Y positions to account for felt at y = -0.6)
    float dHeadRail = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), -TABLE_HALF_WIDTH + CORNER_POCKET_R, TABLE_HALF_WIDTH - CORNER_POCKET_R, -TABLE_HALF_LENGTH, 0);
    float dFootRail = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), -TABLE_HALF_WIDTH + CORNER_POCKET_R, TABLE_HALF_WIDTH - CORNER_POCKET_R, TABLE_HALF_LENGTH, 0);
    float dLeftRailFar = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), SIDE_POCKET_R, TABLE_HALF_LENGTH - CORNER_POCKET_R, -TABLE_HALF_WIDTH, 1);
    float dLeftRailNear = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), -TABLE_HALF_LENGTH + CORNER_POCKET_R, -SIDE_POCKET_R, -TABLE_HALF_WIDTH, 1);
    float dRightRailFar = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), SIDE_POCKET_R, TABLE_HALF_LENGTH - CORNER_POCKET_R, TABLE_HALF_WIDTH, 1);
    float dRightRailNear = sdRailSegmentCorrected(p - float3(0.0, -0.6, 0.0), -TABLE_HALF_LENGTH + CORNER_POCKET_R, -SIDE_POCKET_R, TABLE_HALF_WIDTH, 1);

    float dRails = min(dHeadRail, dFootRail);
    dRails = min(dRails, min(dLeftRailFar, dLeftRailNear));
    dRails = min(dRails, min(dRightRailFar, dRightRailNear));

    // Pocket SDFs (Calculate raw distance to pocket cylinders)
    float dPocketCorners = 1000.0;
    for (int i = 0; i < 4; ++i) dPocketCorners = min(dPocketCorners, sdPocket(p - float3(0.0, -0.6, 0.0), CORNER_POCKET_CENTERS[i], CORNER_POCKET_R));
    float dPocketSides = 1000.0;
    for (int i = 0; i < 2; ++i) dPocketSides = min(dPocketSides, sdPocket(p - float3(0.0, -0.6, 0.0), SIDE_POCKET_CENTERS[i], SIDE_POCKET_R));
    float dPockets = min(dPocketCorners, dPocketSides);
    pocketDist = dPockets; // Store raw pocket distance

    // --- MODIFIED LOGIC ---
    // Find distance to nearest *solid* surface (felt or rail)
    float dSolid = min(dFelt, dRails);

    // Determine preliminary hit type based ONLY on solid surfaces
    if (dSolid == dFelt) {
        hitType = 1.0; // Felt
    } else { // dSolid == dRails
        hitType = 2.0; // Rail
    }
    // NOTE: We no longer use `max(dRails, -dPockets)` here.
    // The check for whether the hit point is *in* a pocket happens later in showScene.

    // Return distance to the nearest solid surface.
    return dSolid;
}

float3 getNormal(float3 p, thread float& hitType, thread float& pocketDist) {
    float2 e = float2(0.0005, 0.0);
    auto map_internal = [&](float3 pt) {
        float ht_ignore, pd_ignore;
        return map(pt, ht_ignore, pd_ignore);
    };
    map(p, hitType, pocketDist);
    return normalize(float3(
        map_internal(p + e.xyy) - map_internal(p - e.xyy),
        map_internal(p + e.yxy) - map_internal(p - e.yxy),
        map_internal(p + e.yyx) - map_internal(p - e.yyx)
    ));
}

// -------------------------------------
//   5) showScene (With Restored Cue Stick)
// -------------------------------------
float3 showScene(float3 ro, float3 rd,
                 float time,
                 float cueOffset,
                 float2 cueTipOffset,
                 constant Ball* balls [[buffer(2)]],
                 int cueVisible,
                 float cueAngle, /* Note: cueAngle is passed but not used in this version of showScene */
                 float2 cue3DRotate)
{
    float3 col = float3(0.05, 0.05, 0.1); // Default Background
    float t = 0.0;
    const float maxDist = 50.0;

    // --- Distance to Balls ---
    float dstBall;
    float3 ballNormal;
    int ballId;
    ballHit(ro, rd, dstBall, ballNormal, ballId, balls);

    // --- Raymarching Setup ---
    float hitDist = maxDist;
    float hitType = 0.0; // 0=miss, 1=felt, 2=rail, 3=pocket_hole, 4=ball, 5=cue
    float pocketDistAtHit = 1000.0; // Store pocket distance at the final hit point
    float3 cueHitPos; // Local coordinates for cue shading

    // --- Raymarching Loop ---
    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * t; // Current point along the ray

        // --- Distance to Cue Stick (Restored Original Logic with Adjusted Height) ---
        float dCueStick = maxDist;
        if (cueVisible != 0 && !isinf(balls[0].velocity.x)) {
            float3 pc = p - float3(balls[0].position.x, balls[0].height + BALL_RADIUS - 0.6, balls[0].position.y);
            // Position the cue stick just above the cue ball
            float baseAngle = sin(time * 0.5) * 0.1;
            pc = rotateX(pc, cue3DRotate.y);
            float finalYaw = baseAngle + cue3DRotate.x;
            pc = rotateY(pc, finalYaw);
            float tipOffset = CUE_LENGTH;
            float maxCueOffset = -BALL_RADIUS;
            pc.z += (maxCueOffset - cueOffset - tipOffset);
            pc = rotateY(pc, PI); // 180-degree rotation to align correctly
            pc.x -= cueTipOffset.x;
            pc.y -= cueTipOffset.y;
            dCueStick = prRoundCylDf(pc, 0.1 - (0.015 / 2.5) * (pc.z + tipOffset), 0.05, CUE_LENGTH);
            cueHitPos = pc;
        }

        // --- Distance to Environment (Felt/Rails - Pockets calculated separately now) ---
        float mapHitType = 0.0; // Preliminary hit type from map (1=felt, 2=rail)
        float mapPocketDist = 1000.0; // Raw distance to pocket cylinders from map
        float dEnv = map(p, mapHitType, mapPocketDist); // Use the corrected map function

        // --- Find Minimum Distance to any object ---
        float d = min(dEnv, dCueStick);

        // --- Hit Detection Logic ---
        if (d < 0.0005 || t > dstBall) { // Potential hit detected (close enough or passed ball intersection)

            if (dstBall <= t + 0.001 && dstBall < maxDist) { // Ray hits a BALL first
                 hitDist = dstBall;
                 hitType = 4.0; // Ball hit type

            } else if (dEnv <= dCueStick) { // Ray hits ENVIRONMENT (Felt or Rail surface) first
                 hitDist = t;

                 // *** Pocket Check at the Precise Hit Point ***
                 // Calculate the distance to the pocket shapes *at the hit point p*
                 float finalPocketDist = 1000.0;
                 float3 p_adjusted = p - float3(0.0, -0.6, 0.0); // Adjust hit point relative to felt plane for pocket SDF

                 // Corner Pockets
                 float finalPocketCorners = 1000.0;
                 for (int k = 0; k < 4; ++k) {
                     finalPocketCorners = min(finalPocketCorners, sdPocket(p_adjusted, CORNER_POCKET_CENTERS[k], CORNER_POCKET_R));
                 }
                 // Side Pockets
                 float finalPocketSides = 1000.0;
                 for (int k = 0; k < 2; ++k) {
                     finalPocketSides = min(finalPocketSides, sdPocket(p_adjusted, SIDE_POCKET_CENTERS[k], SIDE_POCKET_R));
                 }
                 finalPocketDist = min(finalPocketCorners, finalPocketSides);
                 pocketDistAtHit = finalPocketDist; // Store this final pocket distance

                 // Check if the hit point is *inside* a pocket cylinder
                 if (finalPocketDist < -0.01) { // Use a small negative threshold to be safely inside
                     hitType = 3.0; // Yes -> It's a Pocket Hole
                 } else {
                     hitType = mapHitType; // No -> It's the solid surface map detected (Felt=1 or Rail=2)
                 }
                 // *** End Pocket Check ***

            } else if (dCueStick < maxDist) { // Ray hits CUE STICK first
                 hitDist = t;
                 hitType = 5.0; // Cue hit type

            } else { // Ray missed everything or went beyond max distance
                 hitDist = maxDist;
                 hitType = 0.0; // Miss
            }
             break; // Exit the raymarching loop on any hit or miss determination
        } // End Hit Detection

        // --- Advance Ray ---
        // Step forward along the ray direction. Use a safe step size.
        t += max(d * 0.7, 0.001); // 0.7 is a safety factor, 0.001 ensures progress

        // --- Check Max Distance ---
        if (t > maxDist) {
            hitDist = maxDist;
            hitType = 0.0; // Miss
            break; // Exit loop if max distance is exceeded
        }
    } // --- End Raymarching Loop ---

    // --- Shading ---
    if (hitType > 0.0) { // If we hit something (not the background)
        float3 p = ro + rd * hitDist; // Calculate the precise hit point
        float3 n;                     // Normal vector at the hit point
        float3 lightPos = float3(0.0, 20.0, 0.0); // Simple overhead light position
        float3 lightDir = normalize(lightPos - p); // Direction from hit point to light
        float ambient = 0.4;          // Ambient light factor

        // Calculate Normal and Apply Material Shading based on hitType
        if (hitType == 1.0) { // --- Felt Shading ---
            float ignoredHitType; // We know it's felt here
            // Pass pocketDistAtHit, though getNormal won't use it differently for felt vs rail based on map
            n = getNormal(p, ignoredHitType, pocketDistAtHit);
            float2 feltUV = p.xz * 0.5; float feltNoise = fbm(feltUV, 4); float fiberDetail = noise(feltUV*15.0);
            float3 feltBaseColor = float3(0.1, 0.5, 0.2); float3 fiberColor = float3(0.05,0.3,0.1);
            float fiberMix = smoothstep(0.6, 0.8, fiberDetail); float3 feltColor = mix(feltBaseColor, fiberColor, fiberMix);
            feltColor *= (0.8 + 0.2*feltNoise); // Apply larger scale noise variation
            float diff = max(dot(n, lightDir), 0.0); // Diffuse lighting
            float3 r = reflect(rd, n); // Reflection vector for specular
            float spec = pow(max(dot(r, lightDir), 0.0), 8.0); // Specular highlight (low power for felt)
            col = feltColor * (ambient + (1.0-ambient)*diff); // Combine ambient and diffuse
            col += float3(0.05)*spec*(0.5 + 0.5*feltNoise); // Add specular highlight modulated by noise

        } else if (hitType == 2.0) { // --- Rail Shading ---
             float ignoredHitType; // We know it's rail here
             // Pass pocketDistAtHit (which will be >= 0 here)
            n = getNormal(p, ignoredHitType, pocketDistAtHit);
            col = float3(0.4, 0.25, 0.15); // Wood color for rails
            float diff = max(dot(n, lightDir), 0.0); // Diffuse
            float3 r = reflect(rd, n); // Reflection vector
            float spec = pow(max(dot(r, lightDir), 0.0), 32.0); // Sharper specular for wood
            col *= (ambient + (1.0-ambient)*diff); // Combine ambient and diffuse
            col += float3(0.4)*spec; // Add specular highlight

        } else if (hitType == 3.0) { // --- Pocket Hole Shading ---
            col = float3(0.01, 0.01, 0.01); // Very dark color, no lighting needed

        } else if (hitType == 4.0) { // --- Ball Shading ---
            n = ballNormal; // Normal comes directly from ballHit function
            int id = ballId; // Ball ID comes directly from ballHit function

            // Determine Base Color based on Ball ID
            if (id == 0) { col = float3(1.0); } // Cue ball is white
            else {
                bool isStriped = (id >= 9); // Balls 9-15 are striped
                float3 baseColor;
                if (id == 8) { baseColor = float3(0.0); } // 8-ball is black
                else { // Calculate color based on hue for other balls
                    float hue = 0.0;
                    if(id==1 || id==9) hue = 1.0/6.0; // Yellow
                    if(id==2 || id==10) hue = 4.0/6.0; // Blue
                    if(id==3 || id==11) hue = 0.0/6.0; // Red
                    if(id==4 || id==12) hue = 5.0/6.0; // Purple
                    if(id==5 || id==13) hue = 0.5/6.0; // Orange
                    if(id==6 || id==14) hue = 2.0/6.0; // Green
                    if(id==7 || id==15) hue = 0.25/6.0; // Maroon/Brownish
                    baseColor = hsvToRgb(float3(hue, 1.0, 1.0)); // Convert HSV to RGB
                }

                // Apply Rotation using Quaternion
                float3x3 rotMat = qtToRMat(balls[id].quaternion);
                float3 rotatedNormal = rotMat * n; // Rotate the local normal based on ball's orientation

                // Calculate UV coordinates based on rotated normal for patterns
                float2 uv = float2(atan2(rotatedNormal.x, rotatedNormal.z)/(2.0*PI) + 0.5, acos(rotatedNormal.y)/PI);

                // Apply Stripe Pattern
                if (isStriped && id != 8) { // Apply stripe for balls 9-15
                    float stripeWidth = 0.3; // Width of the colored stripe
                    // Mix white and baseColor based on latitude (uv.y)
                    col = mix(float3(1.0), baseColor, step(stripeWidth, uv.y) * step(uv.y, 1.0 - stripeWidth));
                } else {
                    col = baseColor; // Solid balls just use baseColor
                }

                // Apply Number Circle (White Circle)
                float2 circleCenter = float2(0.5, 0.5); // Center of the UV map
                float circleRadius = 0.2; // Radius of the white number circle
                float distToCenter = length(uv - circleCenter);
                if (distToCenter < circleRadius && id != 0) { // If inside circle and not cue ball
                    col = float3(1.0); // Make it white
                }
                // Could add number texture lookup here later using UVs
            }

            // Apply Standard Lighting
            float diff = max(dot(n, lightDir), 0.0); // Diffuse
            col *= (ambient + (1.0-ambient)*diff); // Combine ambient and diffuse
            float3 r = reflect(rd, n); // Reflection vector
            float spec = pow(max(dot(r, lightDir), 0.0), 24.0); // Specular highlight for shiny balls
            col += float3(0.3)*spec; // Add specular

        } else if (hitType == 5.0) { // --- Cue Stick Shading (Restored Original) ---
            float3 eps = float3(0.0005, 0.0, 0.0);
            n = normalize(float3(
                prRoundCylDf(cueHitPos + eps.xyy, 0.1, 0.05, CUE_LENGTH) - prRoundCylDf(cueHitPos - eps.xyy, 0.1, 0.05, CUE_LENGTH),
                prRoundCylDf(cueHitPos + eps.yxy, 0.1, 0.05, CUE_LENGTH) - prRoundCylDf(cueHitPos - eps.yxy, 0.1, 0.05, CUE_LENGTH),
                prRoundCylDf(cueHitPos + eps.yyx, 0.1, 0.05, CUE_LENGTH) - prRoundCylDf(cueHitPos - eps.yyx, 0.1, 0.05, CUE_LENGTH)
            ));
            // Original coloring: different colors near the tip vs butt
            col = (cueHitPos.z < 2.2) ? float3(0.5, 0.3, 0.0) : float3(0.7, 0.7, 0.3);
            float diff = max(dot(n, lightDir), 0.0);
            float3 r = reflect(rd, n);
            float spec = pow(max(dot(r, lightDir), 0.0), 16.0);
            col *= (0.3 + 0.7*diff);
            col += float3(0.2)*spec;
        }

    } else { // --- Background Shading ---
        // Ray missed everything, use the default background color
        col = float3(0.05, 0.05, 0.1);
        // Could add a sky gradient or stars here
        // float sky = max(rd.y, 0.0);
        // col = mix(float3(0.1, 0.1, 0.2), float3(0.3, 0.4, 0.6), sky);
    }

    // Final Color Clamping
    return clamp(col, 0.0, 1.0); // Ensure color values are within valid range
}

// -------------------------------------
//   6) Vertex & Fragment Shaders
// -------------------------------------
vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[4] = { float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0), float2( 1.0,  1.0) };
    constexpr float2 uvs[4] = { float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0), float2(1.0, 1.0) };
    VertexOut out; out.position = float4(positions[vertexID], 0.0, 1.0); out.uv = uvs[vertexID]; return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant float2 &resolution   [[buffer(0)]],
                               constant float &time          [[buffer(1)]],
                               constant Ball* balls          [[buffer(2)]],
                               constant float &cueOffset     [[buffer(3)]],
                               constant int &cueVisible      [[buffer(4)]],
                               constant float2 &cueTipOffset [[buffer(5)]],
                               constant float &cueAngle      [[buffer(6)]],
                               constant float2 &cue3DRotate  [[buffer(7)]]) {
    float2 uv = 2.0 * in.uv - 1.0; uv.x *= resolution.x / resolution.y;
    float angle = time * 0.1;
    float3 camPos = float3(sin(angle)*25.0, 12.0, cos(angle)*25.0);
    float3 camTarget = float3(0.0, -0.6, 0.0); // Adjusted to look at the new felt height
    float3 ww = normalize(camTarget - camPos);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.7;
    float3 rd = normalize(ww + uu*uv.x*fov + vv*uv.y*fov);
    float3 col = showScene(camPos, rd, time, cueOffset, cueTipOffset, balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}

fragment float4 behindBallFragmentShader(VertexOut in [[stage_in]],
                                         constant float2 &resolution   [[buffer(0)]],
                                         constant float3 &cameraPos    [[buffer(1)]],
                                         constant float3 &cameraTarget [[buffer(2)]],
                                         constant Ball*  balls         [[buffer(3)]],
                                         constant float  &cueOffset    [[buffer(4)]],
                                         constant int    &cueVisible   [[buffer(5)]],
                                         constant float2 &cueTipOffset [[buffer(6)]],
                                         constant float  &cueAngle     [[buffer(7)]],
                                         constant float2 &cue3DRotate  [[buffer(8)]]) {
    float2 uv = 2.0 * in.uv - 1.0; uv.x *= resolution.x / resolution.y;
    float3 ro = cameraPos; float3 target = cameraTarget;
    float3 ww = normalize(target - ro);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.8;
    float3 rd = normalize(ww + uu*uv.x*fov + vv*uv.y*fov);
    float timeDummy = 0.0;
    float3 col = showScene(ro, rd, timeDummy, cueOffset, cueTipOffset, balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}

fragment float4 thirdBallFragmentShader(VertexOut in [[stage_in]],
                                        constant float2 &resolution   [[buffer(0)]],
                                        constant float3 &cameraPos    [[buffer(1)]],
                                        constant float3 &cameraTarget [[buffer(2)]],
                                        constant Ball*  balls         [[buffer(3)]],
                                        constant float  &cueOffset    [[buffer(4)]],
                                        constant int    &cueVisible   [[buffer(5)]],
                                        constant float2 &cueTipOffset [[buffer(6)]],
                                        constant float  &cueAngle     [[buffer(7)]],
                                        constant float2 &cue3DRotate  [[buffer(8)]]) {
    float2 uv = 2.0 * in.uv - 1.0; uv.x *= resolution.x / resolution.y;
    float3 ro = cameraPos; float3 target = cameraTarget;
    float3 ww = normalize(target - ro);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(ww, uu));
    const float fov = 0.8;
    float3 rd = normalize(ww + uu*uv.x*fov + vv*uv.y*fov);
    float timeDummy = 0.0;
    float3 col = showScene(ro, rd, timeDummy, cueOffset, cueTipOffset, balls, cueVisible, cueAngle, cue3DRotate);
    return float4(col, 1.0);
}
"""

// MARK: - Swift Utility Functions
func quaternionFromAxisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
    let halfAngle = angle * 0.5
    let s = sin(halfAngle)
    return SIMD4<Float>(axis.x * s, axis.y * s, axis.z * s, cos(halfAngle))
}

func quaternionMultiply(_ q1: SIMD4<Float>, _ q2: SIMD4<Float>) -> SIMD4<Float> {
    SIMD4<Float>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    )
}

// MARK: - Ball Data
struct BallData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var height: Float = 0.01
    var verticalVelocity: Float = 0.0
    var angularVelocity: SIMD3<Float>
    var quaternion: SIMD4<Float>
}

struct BallShaderData {
    var position: SIMD2<Float>
    var velocity: SIMD2<Float>
    var quaternion: SIMD4<Float>
    var height: Float
}

// MARK: - BilliardSimulation Class
final class BilliardSimulation: ObservableObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private let orbitPipeline: MTLRenderPipelineState
    private let behindPipeline: MTLRenderPipelineState
    private let thirdPipeline: MTLRenderPipelineState

    @Published var time: Float = 0.0
    @Published var isTouching: Bool = false
    @Published var cueOffset: Float = 0.0
    @Published var cueTipOffset: SIMD2<Float> = .zero
    @Published var showCueValue: Int32 = 1
    @Published var cueAngle: Float = 0.0
    @Published var cue3DRotate: SIMD2<Float> = SIMD2<Float>(0, 0)
    @Published var shooting: Bool = false

    var balls: [BallData]
    private var ballBuffer: MTLBuffer

    // Various table and physics definitions:
    private let ballRadius: Float = 0.47
    private let tableWidth: Float = 7.6
    private let tableLength: Float = 13.6
    private let pocketRadius: Float = 0.53  // Not used for pockets anymore—kept for legacy
    private let cushionEdgeX: Float = 7.6
    private let cushionEdgeZ: Float = 13.6
    private let cuePullSpeed: Float = 1.0
    private let cueStrikeSpeed: Float = 5.0
    private let maxCueOffset: Float = 2.0
    public let maxTipOffset: Float = 0.47
    private let ballMass: Float = 0.17
    private let momentOfInertia: Float = 0.4 * 0.17 * 0.47 * 0.47
    private let gravity: Float = 9.81
    private let frictionKinetic: Float = 0.25
    private let frictionRolling: Float = 0.015
    private let frictionSpinDecay: Float = 8.0
    private let restitutionBall: Float = 0.95
    private let restitutionCushion: Float = 0.8
    private let ballFriction: Float = 0.05

    private var hitTriggered: Bool = false
    private var powerAtRelease: Float = 0.0

    private var resolution = SIMD2<Float>(0, 0)
    private var orbitUniformsBuffer: MTLBuffer
    private var behindCamPosBuffer: MTLBuffer
    private var behindCamTargetBuffer: MTLBuffer
    private var thirdCamPosBuffer: MTLBuffer
    private var thirdCamTargetBuffer: MTLBuffer
    private var cueOffsetBuffer: MTLBuffer
    private var showCueBuffer: MTLBuffer
    private var cueTipOffsetBuffer: MTLBuffer
    private var cueAngleBuffer: MTLBuffer
    private var cue3DRotateBuffer: MTLBuffer
    private let tableHalfLength: Float = 13.6
    
    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let cq = dev.makeCommandQueue() else {
            return nil
        }
        device = dev
        commandQueue = cq

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalShader, options: nil)
        } catch {
            print("Failed to create Metal library: \(error)")
            return nil
        }

        guard let vertexFunction = library.makeFunction(name: "vertexShader"),
              let orbitFragment = library.makeFunction(name: "fragmentShader"),
              let behindFragment = library.makeFunction(name: "behindBallFragmentShader"),
              let thirdFragment = library.makeFunction(name: "thirdBallFragmentShader")
        else {
            print("Missing required shader functions.")
            return nil
        }

        do {
            let orbitDescriptor = MTLRenderPipelineDescriptor()
            orbitDescriptor.vertexFunction = vertexFunction
            orbitDescriptor.fragmentFunction = orbitFragment
            orbitDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            orbitPipeline = try device.makeRenderPipelineState(descriptor: orbitDescriptor)

            let behindDescriptor = MTLRenderPipelineDescriptor()
            behindDescriptor.vertexFunction = vertexFunction
            behindDescriptor.fragmentFunction = behindFragment
            behindDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            behindPipeline = try device.makeRenderPipelineState(descriptor: behindDescriptor)

            let thirdDescriptor = MTLRenderPipelineDescriptor()
            thirdDescriptor.vertexFunction = vertexFunction
            thirdDescriptor.fragmentFunction = thirdFragment
            thirdDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            thirdPipeline = try device.makeRenderPipelineState(descriptor: thirdDescriptor)
        } catch {
            print("Failed to create pipeline state: \(error)")
            return nil
        }

        let r: Float = 0.47  // Ball radius
        let d: Float = 2.0 * r  // Ball diameter = 0.94
        let sqrt3_2: Float = sqrt(3.0) / 2.0  // ≈ 0.866
        let rowSpacing: Float = d * sqrt3_2  // Vertical spacing between rows ≈ 0.814
        let headSpotZ: Float = tableHalfLength / 2.0  // z = 6.8 (quarter of table length from head rail)
        let footSpotZ: Float = -tableHalfLength / 2.0  // z = -6.8 (center of the rack, same distance from center as head spot)
        let identityQuat = SIMD4<Float>(0, 0, 0, 1)
        self.balls = [
            // Cue ball (index 0) at head spot
            BallData(position: SIMD2<Float>(0.0, headSpotZ), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            // Row 1 (1 ball, 2 rows above 8-ball, index 1)
            BallData(position: SIMD2<Float>(0.0, footSpotZ + 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            // Row 2 (2 balls, 1 row above 8-ball, indices 2–3)
            BallData(position: SIMD2<Float>(-d / 2.0, footSpotZ + rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(d / 2.0, footSpotZ + rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            // Row 3 (3 balls, same z as 8-ball, indices 4–6, 8-ball at index 8)
            BallData(position: SIMD2<Float>(-d, footSpotZ), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 4
            BallData(position: SIMD2<Float>(-1.5 * d, footSpotZ - rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 5, moved to row 4
            BallData(position: SIMD2<Float>(d, footSpotZ), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 6
            // Row 4 (4 balls, 1 row below 8-ball, indices 7–10)
            BallData(position: SIMD2<Float>(-0.5 * d, footSpotZ - rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 7
            // 8-ball (index 8) at the center of the rack (center of row 3)
            BallData(position: SIMD2<Float>(0.0, footSpotZ), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.5 * d, footSpotZ - rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 9
            BallData(position: SIMD2<Float>(1.5 * d, footSpotZ - rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat), // Index 10
            // Row 5 (5 balls, 2 rows below 8-ball, indices 11–15)
            BallData(position: SIMD2<Float>(-2.0 * d, footSpotZ - 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(-1.0 * d, footSpotZ - 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(0.0, footSpotZ - 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(1.0 * d, footSpotZ - 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat),
            BallData(position: SIMD2<Float>(2.0 * d, footSpotZ - 2.0 * rowSpacing), velocity: .zero, angularVelocity: .zero, quaternion: identityQuat)
        ]

        var ballShaderData = [BallShaderData](
            repeating: BallShaderData(position: .zero, velocity: .zero, quaternion: .zero, height: 0.0),
            count: 16
        )
        for i in 0..<16 {
            ballShaderData[i] = BallShaderData(
                position: balls[i].position,
                velocity: balls[i].velocity,
                quaternion: balls[i].quaternion,
                height: balls[i].height
            )
        }
        self.ballBuffer = device.makeBuffer(
            bytes: ballShaderData,
            length: MemoryLayout<BallShaderData>.stride * 16,
            options: .storageModeShared
        )!

        orbitUniformsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        behindCamPosBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        behindCamTargetBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamPosBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        thirdCamTargetBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        cueOffsetBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        showCueBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        cueTipOffsetBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        cueAngleBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        cue3DRotateBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
    }

    private func checkPocket(pos: SIMD2<Float>, height: Float) -> Bool {
        // This method is replaced by real pockets in the SDF, but we keep it for demonstration.
        let pocketPositions: [SIMD2<Float>] = [
            SIMD2<Float>(-8,14),  SIMD2<Float>( 8,14),
            SIMD2<Float>(-8, 0),  SIMD2<Float>( 8, 0),
            SIMD2<Float>(-8,-14), SIMD2<Float>( 8,-14)
        ]
        for p in pocketPositions {
            if simd_length(pos - p) < pocketRadius && height <= 0.01 + ballRadius {
                return true
            }
        }
        return false
    }

    func updatePhysics(deltaTime: Float) {
        if isTouching && !shooting {
            cueOffset += cuePullSpeed * deltaTime
            if cueOffset > maxCueOffset { cueOffset = maxCueOffset }
        } else if !isTouching && cueOffset > 0.0 {
            if !shooting {
                powerAtRelease = cueOffset / maxCueOffset
                shooting = true
            }
            cueOffset -= cueStrikeSpeed * deltaTime
            if cueOffset <= 0.0 {
                cueOffset = 0.0
                applyCueStrike()
                shooting = false
                hitTriggered = true
                showCueValue = 0
            }
        }

        if !isTouching && hitTriggered {
            var allStopped = true
            for ball in balls {
                if simd_length(ball.velocity) > 0.01
                   || abs(ball.verticalVelocity) > 0.01
                   || simd_length(ball.angularVelocity) > 0.05 {
                    allStopped = false
                    break
                }
            }
            if allStopped {
                hitTriggered = false
                showCueValue = 1
            }
        }

        let subSteps = 8
        let dt = deltaTime / Float(subSteps)

        for _ in 0..<subSteps {
            for i in 0..<16 {
                var ball = balls[i]
                if ball.velocity.x.isInfinite { continue }

                let v = ball.velocity
                let w = ball.angularVelocity

                // Gravity
                ball.verticalVelocity -= gravity * dt
                ball.height += ball.verticalVelocity * dt
                if ball.height <= 0.01 {
                    ball.height = 0.01
                    if ball.verticalVelocity < 0 {
                        ball.verticalVelocity = -ball.verticalVelocity * restitutionCushion
                        if abs(ball.verticalVelocity) < 0.1 { ball.verticalVelocity = 0.0 }
                    }
                }

                // Friction if on the table
                if ball.height <= 0.011 {
                    // Compute the velocity of the contact point by combining linear velocity (v) with rotation (w)
                    let r3D = SIMD3<Float>(0, -ballRadius, 0)
                    let v3D = SIMD3<Float>(v.x, 0, v.y)
                    let contactVel3D = v3D + simd_cross(w, r3D)
                    let relativeVelocityAtContact = SIMD2<Float>(contactVel3D.x, contactVel3D.z)
                    let sliding = simd_length(relativeVelocityAtContact) > 0.02

                    if sliding {
                        let frictionDir = -simd_normalize(relativeVelocityAtContact)
                        let frictionForce = frictionKinetic * ballMass * gravity
                        let accel = frictionForce / ballMass * frictionDir
                        ball.velocity += accel * dt

                        let torque = frictionForce * ballRadius
                        let alpha = torque / momentOfInertia * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                        ball.angularVelocity += alpha * dt
                    } else {
                        let vMag = simd_length(v)
                        if vMag > 0 {
                            let frictionDir = -simd_normalize(v)
                            let frictionForce = frictionRolling * ballMass * gravity
                            let accel = frictionForce / ballMass * frictionDir
                            ball.velocity += accel * dt

                            let alpha = frictionForce / (ballRadius * momentOfInertia)
                                * SIMD3<Float>(-frictionDir.y, 0, frictionDir.x)
                            ball.angularVelocity += alpha * dt
                        }
                    }

                    let wMag = simd_length(ball.angularVelocity)
                    if wMag > 0 {
                        let decay = -simd_normalize(ball.angularVelocity) * frictionSpinDecay * dt
                        ball.angularVelocity += decay
                        if simd_dot(ball.angularVelocity + decay, ball.angularVelocity) <= 0 {
                            ball.angularVelocity = .zero
                        }
                    }
                }

                ball.position += ball.velocity * dt

                // Update orientation from spin:
                let wMag2 = simd_length(ball.angularVelocity)
                if wMag2 > 0 {
                    let axis = ball.angularVelocity / wMag2
                    let angle = wMag2 * dt
                    let deltaQuat = quaternionFromAxisAngle(axis, angle)
                    ball.quaternion = quaternionMultiply(deltaQuat, ball.quaternion)
                    ball.quaternion = simd_normalize(ball.quaternion)
                }

                // Simple cushion collisions:
                if abs(ball.position.x) > cushionEdgeX - ballRadius && ball.height <= 0.01 + ballRadius {
                    ball.position.x = (ball.position.x > 0)
                        ? (cushionEdgeX - ballRadius)
                        : -(cushionEdgeX - ballRadius)
                    ball.velocity.x = -ball.velocity.x * restitutionCushion
                    let spinChange = -ball.angularVelocity.z * 0.5
                    ball.angularVelocity.z += spinChange
                    ball.angularVelocity.y *= 0.6
                }
                if abs(ball.position.y) > cushionEdgeZ - ballRadius && ball.height <= 0.01 + ballRadius {
                    ball.position.y = (ball.position.y > 0)
                        ? (cushionEdgeZ - ballRadius)
                        : -(cushionEdgeZ - ballRadius)
                    ball.velocity.y = -ball.velocity.y * restitutionCushion
                    let spinChange = ball.angularVelocity.x * 0.5
                    ball.angularVelocity.x += spinChange
                    ball.angularVelocity.y *= 0.6
                }

                // Pocket check if near slate drop
                if (i != 0 || ball.height <= 0.01 + ballRadius)
                   && checkPocket(pos: ball.position, height: ball.height) {
                    // Mark as pocketed:
                    ball.velocity = SIMD2<Float>(.infinity, .infinity)
                    ball.verticalVelocity = 0.0
                    ball.angularVelocity = .zero
                    ball.position = .zero
                    ball.height = 0.01
                }

                balls[i] = ball
            }

            // Ball-ball collisions:
            for i in 0..<15 {
                for j in (i+1)..<16 {
                    var b1 = balls[i]
                    var b2 = balls[j]
                    if b1.velocity.x.isInfinite || b2.velocity.x.isInfinite { continue }

                    let delta = b2.position - b1.position
                    let dist = simd_length(delta)
                    let heightDiff = abs(b1.height - b2.height)

                    if dist < 2.0 * ballRadius && dist > 0
                       && (heightDiff < ballRadius
                           || (b1.height <= 0.01 + ballRadius && b2.height <= 0.01 + ballRadius)) {
                        let normal = delta / dist
                        let relVel = b1.velocity - b2.velocity
                        let impulse = simd_dot(relVel, normal)
                        if impulse > 0 {
                            let impulseMag = -(1.0 + restitutionBall) * impulse / (2.0 / ballMass)
                            let impulseVec = normal * impulseMag
                            b1.velocity += impulseVec / ballMass
                            b2.velocity -= impulseVec / ballMass

                            let tangent = SIMD2<Float>(-normal.y, normal.x)
                            let relVelTangent = simd_dot(relVel, tangent)
                            let frictionImpulse = min(ballFriction * abs(impulseMag),
                                                      abs(relVelTangent) * ballMass)
                            let frictionVec = tangent * frictionImpulse * (relVelTangent > 0 ? -1 : 1)
                            b1.velocity += frictionVec / ballMass
                            b2.velocity -= frictionVec / ballMass

                            let spinChange = frictionImpulse / (ballRadius * momentOfInertia)
                            b1.angularVelocity += SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange
                            b2.angularVelocity -= SIMD3<Float>(-tangent.y, 0, tangent.x) * spinChange

                            let overlap = 2.0 * ballRadius - dist
                            let correction = normal * (overlap * 0.5)
                            b1.position -= correction
                            b2.position += correction
                        }
                    }
                    balls[i] = b1
                    balls[j] = b2
                }
            }

            // Kill very small movement:
            for i in 0..<16 {
                let vMag = simd_length(balls[i].velocity)
                let wMag = simd_length(balls[i].angularVelocity)
                if vMag < 0.01
                   && abs(balls[i].verticalVelocity) < 0.01
                   && wMag < 0.05 {
                    balls[i].velocity = .zero
                    balls[i].verticalVelocity = 0.0
                    balls[i].angularVelocity = .zero
                }
            }
        }

        // Copy updated ball data to GPU buffer
        let ptr = ballBuffer.contents().bindMemory(to: BallShaderData.self, capacity: 16)
        for i in 0..<16 {
            ptr[i] = BallShaderData(
                position: balls[i].position,
                velocity: balls[i].velocity,
                quaternion: balls[i].quaternion,
                height: balls[i].height
            )
        }
    }

    func rotateX(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let c = cos(angle)
        let s = sin(angle)
        return SIMD3<Float>(
            vector.x,
            vector.y * c - vector.z * s,
            vector.y * s + vector.z * c
        )
    }

    func rotateY(_ vector: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
        let c = cos(angle)
        let s = sin(angle)
        return SIMD3<Float>(
            vector.x * c + vector.z * s,
            vector.y,
            -vector.x * s + c * vector.z
        )
    }

    private func applyCueStrike() {
        var cueDir = SIMD3<Float>(0,0,-1)
        cueDir = rotateX(cueDir, cue3DRotate.y)
        cueDir = rotateY(cueDir, -cue3DRotate.x)
        let baseSpeed: Float = 15.0
        let velocityScale = 0.5 + 1.5 * powerAtRelease

        let tipOffset3D = SIMD3<Float>(cueTipOffset.x, -cueTipOffset.y, 0)
        let spinFactor: Float = 10.0 / (2.0 * ballRadius)
        let angularVelocity = simd_cross(cueDir, tipOffset3D) * spinFactor * velocityScale
        balls[0].angularVelocity = angularVelocity

        let jumpFactor = -sin(cue3DRotate.y) * baseSpeed * velocityScale
        balls[0].verticalVelocity = jumpFactor > 0 ? jumpFactor : 0.0

        let spinEffect = simd_cross(angularVelocity, SIMD3<Float>(cueDir.x, 0, cueDir.z)) * 0.3
        let adjustedDir = simd_normalize(cueDir + spinEffect)
        let adjustedDir2D = simd_normalize(SIMD2<Float>(adjustedDir.x, adjustedDir.z))
        balls[0].velocity = adjustedDir2D * (baseSpeed * velocityScale)

        powerAtRelease = 0.0
    }

    func encodeOrbitRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))
        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

        let timePtr = orbitUniformsBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        timePtr[0] = self.time
        encoder.setFragmentBuffer(orbitUniformsBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 2)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 3)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 4)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 5)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 6)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 7)

        encoder.setRenderPipelineState(orbitPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func encodeBehindRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0,2.0,0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, whiteBall.height, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        // If the white ball is basically stopped, place the camera a bit behind it:
        if speed < 0.01 {
            let stationaryDistance: Float = 2.5
            var offset = SIMD3<Float>(0,0,stationaryDistance)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset
            cameraPosition.y = 0.7
        } else {
            // If it's moving, track behind it:
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x,0,whiteBall.velocity.y))
            cameraPosition = cameraTarget - forward*3.0
            cameraPosition.y += 1.0
        }

        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

        behindCamPosBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraPosition
        encoder.setFragmentBuffer(behindCamPosBuffer, offset: 0, index: 1)

        behindCamTargetBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraTarget
        encoder.setFragmentBuffer(behindCamTargetBuffer, offset: 0, index: 2)

        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 3)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 4)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 5)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 6)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 7)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 8)

        encoder.setRenderPipelineState(behindPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    func encodeThirdRenderPass(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        self.resolution = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))

        let whiteBall = balls[0]
        var cameraPosition = SIMD3<Float>(0,2.0,0)
        var cameraTarget = SIMD3<Float>(whiteBall.position.x, whiteBall.height, whiteBall.position.y)
        let speed = simd_length(whiteBall.velocity)

        if speed < 0.01 {
            let stationaryDistance: Float = 7.0
            var offset = SIMD3<Float>(0,0,stationaryDistance)
            offset = rotateX(offset, cue3DRotate.y)
            offset = rotateY(offset, -cue3DRotate.x)
            cameraPosition = cameraTarget + offset
            let cueBaseHeight: Float = 0.01
            let cueAngleVertical = cue3DRotate.y
            let verticalAdjustment = sin(cueAngleVertical)*stationaryDistance
            cameraPosition.y = cueBaseHeight + verticalAdjustment + 0.7
        } else {
            let forward = simd_normalize(SIMD3<Float>(whiteBall.velocity.x,0,whiteBall.velocity.y))
            cameraPosition = cameraTarget - forward*8.0
            cameraPosition.y += 1.0
        }

        encoder.setFragmentBytes(&resolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

        thirdCamPosBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraPosition
        encoder.setFragmentBuffer(thirdCamPosBuffer, offset: 0, index: 1)

        thirdCamTargetBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)[0] = cameraTarget
        encoder.setFragmentBuffer(thirdCamTargetBuffer, offset: 0, index: 2)

        encoder.setFragmentBuffer(ballBuffer, offset: 0, index: 3)

        cueOffsetBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueOffset
        encoder.setFragmentBuffer(cueOffsetBuffer, offset: 0, index: 4)

        showCueBuffer.contents().bindMemory(to: Int32.self, capacity: 1)[0] = showCueValue
        encoder.setFragmentBuffer(showCueBuffer, offset: 0, index: 5)

        cueTipOffsetBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cueTipOffset
        encoder.setFragmentBuffer(cueTipOffsetBuffer, offset: 0, index: 6)

        cueAngleBuffer.contents().bindMemory(to: Float.self, capacity: 1)[0] = cueAngle
        encoder.setFragmentBuffer(cueAngleBuffer, offset: 0, index: 7)

        cue3DRotateBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0] = cue3DRotate
        encoder.setFragmentBuffer(cue3DRotateBuffer, offset: 0, index: 8)

        encoder.setRenderPipelineState(thirdPipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}

// MARK: - SwiftUI Views
struct OrbitingMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: OrbitingMetalView
        init(_ parent: OrbitingMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }
            parent.simulation.encodeOrbitRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

struct BehindBallMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: BehindBallMetalView
        init(_ parent: BehindBallMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }
            parent.simulation.encodeBehindRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

struct ThirdBallMetalView: UIViewRepresentable {
    @ObservedObject var simulation: BilliardSimulation

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: ThirdBallMetalView
        init(_ parent: ThirdBallMetalView) { self.parent = parent }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let commandBuffer = parent.simulation.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
            else { return }
            parent.simulation.encodeThirdRenderPass(encoder: encoder, viewSize: view.drawableSize)
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: simulation.device)
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        return mtkView
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// MARK: - Main ContentView
struct ContentView: View {
    @StateObject private var simulation = BilliardSimulation()!
    @State private var viewSizeBehind: CGSize = .zero
    @State private var initialTouchBehind: CGPoint? = nil
    @State private var initialTipOffsetBehind: SIMD2<Float> = .zero

    @State private var viewSizeThird: CGSize = .zero
    @State private var initialTouchThird: CGPoint? = nil
    @State private var initialCueYawThird: Float = 0.0
    @State private var initialCuePitchThird: Float = 0.0

    var body: some View {
        ZStack {
            OrbitingMetalView(simulation: simulation)
                .edgesIgnoringSafeArea(.all)
                .overlay(Text("Orbiting Camera")
                    .foregroundColor(.white)
                    .padding(),
                         alignment: .top)
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in simulation.isTouching = true }
                        .onEnded { _ in simulation.isTouching = false }
                )

            VStack {
                Spacer()
                HStack {
                    BehindBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width/2,
                               height: UIScreen.main.bounds.width/2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeBehind = geo.size }
                                    .onChange(of: geo.size) { newSize in viewSizeBehind = newSize }
                            }
                        )
                        .overlay(Text("Behind-Ball Camera")
                            .foregroundColor(.white)
                            .padding(),
                                 alignment: .top)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if simulation.showCueValue == 1 && !simulation.shooting {
                                        if initialTouchBehind == nil {
                                            initialTouchBehind = value.startLocation
                                            initialTipOffsetBehind = simulation.cueTipOffset
                                        }
                                        guard let start = initialTouchBehind else { return }
                                        let deltaX = Float(value.location.x - start.x)
                                        let deltaY = Float(value.location.y - start.y)
                                        let scaleFactor = simulation.maxTipOffset / Float(viewSizeBehind.height) * 2.0
                                        let aspect = Float(viewSizeBehind.width / viewSizeBehind.height)
                                        var newOffset = initialTipOffsetBehind + SIMD2<Float>(
                                            deltaX * scaleFactor * aspect,
                                            -deltaY * scaleFactor
                                        )
                                        let offsetLength = simd_length(newOffset)
                                        if offsetLength > simulation.maxTipOffset {
                                            newOffset *= simulation.maxTipOffset / offsetLength
                                        }
                                        simulation.cueTipOffset = newOffset
                                    }
                                }
                                .onEnded { _ in
                                    initialTouchBehind = nil
                                    initialTipOffsetBehind = .zero
                                }
                        )

                    ThirdBallMetalView(simulation: simulation)
                        .frame(width: UIScreen.main.bounds.width/2,
                               height: UIScreen.main.bounds.width/2)
                        .background(
                            GeometryReader { geo in
                                Color.clear
                                    .onAppear { viewSizeThird = geo.size }
                                    .onChange(of: geo.size) { newSize in viewSizeThird = newSize }
                            }
                        )
                        .overlay(Text("Third-Ball Camera (3D Cue Rotation)")
                            .foregroundColor(.white)
                            .padding(),
                                 alignment: .top)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if simulation.showCueValue == 1 && !simulation.shooting {
                                        if initialTouchThird == nil {
                                            initialTouchThird = value.startLocation
                                            initialCueYawThird = simulation.cue3DRotate.x
                                            initialCuePitchThird = simulation.cue3DRotate.y
                                        }
                                        guard let start = initialTouchThird else { return }
                                        let deltaX = Float(value.location.x - start.x)
                                        let deltaY = Float(value.location.y - start.y)
                                        let sensitivity: Float = 0.01
                                        let newYaw = initialCueYawThird - deltaX * sensitivity
                                        let newPitch = initialCuePitchThird - deltaY * sensitivity
                                        let clampedPitch = max(-0.8, min(0.8, newPitch))
                                        simulation.cue3DRotate = SIMD2<Float>(newYaw, clampedPitch)
                                    }
                                }
                                .onEnded { _ in initialTouchThird = nil }
                        )
                }
                .padding(.bottom, 20)
            }
        }
        .onAppear {
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { _ in
                simulation.time += 1.0/60.0
                simulation.updatePhysics(deltaTime: 1.0/60.0)
            }
            RunLoop.current.add(timer, forMode: .common)
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
