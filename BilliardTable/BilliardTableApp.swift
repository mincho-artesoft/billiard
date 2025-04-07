import SwiftUI
import MetalKit
import simd // For matrix math

// MARK: - Constants and Parameters
enum Constants {
    // --- Table Dimensions (Meters) --- Conversion from mm
    static let tableLength: Float = 2.7432  // 9ft
    static let tableWidth: Float = 1.3716   // 4.5ft
    static let tableHeight: Float = 0.800    // Height to top of rail
    // Adjust playing surface elevation based on K55 height
    static let k55_SubrailHeight: Float = 1.688 * 0.0254 // ≈ 0.04288m (Relative to playing surface if playing surface was 0)
    static let k55_NoseHeight: Float = 1.458 * 0.0254    // ≈ 0.03703m (Relative to playing surface if playing surface was 0)
    static let playingSurfaceElevation: Float = tableHeight - k55_SubrailHeight // Align table surface with bottom of cushion subrail support

    // --- Rails and Cushions (Meters) ---
    static let railWidth: Float = 0.125 // Width of the wooden rail part
    static let cushionAngleDegrees: Float = 23.5 // Angle of subrail face from vertical

    // --- Pockets (Meters) ---
    static let cornerPocketOpening: Float = 0.115 // Mouth width
    static let sidePocketOpening: Float = 0.130   // Mouth width
    static let pocketDepth: Float = 0.100

    // --- Materials ---
    static let feltColor: SIMD3<Float> = SIMD3<Float>(34/255.0, 139/255.0, 34/255.0)
    static let woodColor: SIMD3<Float> = SIMD3<Float>(0.3, 0.15, 0.08)
    static let leatherColor: SIMD3<Float> = SIMD3<Float>(0.1, 0.08, 0.05)
    static let markerColor: SIMD3<Float> = SIMD3<Float>(0.9, 0.9, 0.9)

    // --- Material Properties (PBR) ---
    static let feltRoughness: Float = 0.8; static let feltMetallic: Float = 0.01; static let feltSpecular: Float = 0.1
    static let woodRoughness: Float = 0.3; static let woodMetallic: Float = 0.0; static let woodSpecular: Float = 0.5
    static let leatherRoughness: Float = 0.6; static let leatherMetallic: Float = 0.0; static let leatherSpecular: Float = 0.2
    static let cushionRoughness: Float = 0.75; static let cushionMetallic: Float = 0.01; static let cushionSpecular: Float = 0.1
    static let markerRoughness: Float = 0.4; static let markerMetallic: Float = 0.1; static let markerSpecular: Float = 0.5

    // --- Rendering ---
    static let markerRadius: Float = 0.01
    static let shadowMapSize: Int = 1024 // Keep reduced size for stability
}

// MARK: - Data Structures
struct Vertex { var position: SIMD3<Float>; var normal: SIMD3<Float>; var texCoords: SIMD2<Float> }
struct Uniforms { var modelMatrix: float4x4; var viewMatrix: float4x4; var projectionMatrix: float4x4; var normalMatrix: float3x3; var cameraWorldPosition: SIMD3<Float>; var lightWorldPosition: SIMD3<Float>; var lightViewMatrix: float4x4; var lightProjectionMatrix: float4x4 }
struct Material { var baseColor: SIMD3<Float>; var roughness: Float; var metallic: Float; var specular: Float; var ambientOcclusion: Float = 1.0; var usesTexture: Bool = false }
struct Mesh { var vertexBuffer: MTLBuffer; var indexBuffer: MTLBuffer; var indexCount: Int; var primitiveType: MTLPrimitiveType; var material: Material; var texture: MTLTexture? = nil }

// MARK: - Math Utilities
let matrix_identity_float4x4=float4x4(1); let matrix_identity_float3x3=float3x3(1)
func matrix_float4x4_translation(_ t: SIMD3<Float>) -> float4x4 { return float4x4([1,0,0,0],[0,1,0,0],[0,0,1,0],[t.x,t.y,t.z,1])}
func matrix_float4x4_scaling(_ s: SIMD3<Float>) -> float4x4 { return float4x4([s.x,0,0,0],[0,s.y,0,0],[0,0,s.z,0],[0,0,0,1])}
func matrix_float4x4_rotation(radians: Float, axis: SIMD3<Float>) -> float4x4 { let u=normalize(axis);let ct=cosf(radians);let st=sinf(radians);let ci=1-ct;let x=u.x,y=u.y,z=u.z;return float4x4([ct+x*x*ci,y*x*ci+z*st,z*x*ci-y*st,0],[x*y*ci-z*st,ct+y*y*ci,z*y*ci+x*st,0],[x*z*ci+y*st,y*z*ci-x*st,ct+z*z*ci,0],[0,0,0,1])}
func matrix_perspective_right_hand(fovyRadians f:Float,aspect a:Float,nearZ n:Float,farZ fz:Float)->float4x4{let ys=1/tanf(f*0.5);let xs=ys/a;let zs=fz/(n-fz);return float4x4([xs,0,0,0],[0,ys,0,0],[0,0,zs,-1],[0,0,zs*n,0])}
func matrix_look_at_right_hand(eye e:SIMD3<Float>,center c:SIMD3<Float>,up u:SIMD3<Float>)->float4x4{let z=normalize(e-c);let x=normalize(cross(u,z));let y=cross(z,x);let t=SIMD3<Float>(-dot(x,e),-dot(y,e),-dot(z,e));return float4x4([x.x,y.x,z.x,0],[x.y,y.y,z.y,0],[x.z,y.z,z.z,0],[t.x,t.y,t.z,1])}
func matrix_ortho_right_hand(left L:Float,right R:Float,bottom B:Float,top T:Float,nearZ N:Float,farZ F:Float)->float4x4{return float4x4([2/(R-L),0,0,0],[0,2/(T-B),0,0],[0,0,-1/(F-N),0],[-(R+L)/(R-L),-(T+B)/(T-B),-N/(F-N),1])}

// MARK: - Geometry Generation (Corrected Initializers)
class GeometryGenerator {
    let device: MTLDevice
    init(device: MTLDevice) { self.device = device }

    private func createBuffers(vertices: [Vertex], indices: [UInt32]) -> (MTLBuffer, MTLBuffer, Int) {
        guard !vertices.isEmpty, !indices.isEmpty else { print("Warn: Empty geom"); let minSize=MemoryLayout<Float>.stride; let vb=device.makeBuffer(length:max(minSize,1),options:.storageModeShared)!; let ib=device.makeBuffer(length:max(MemoryLayout<UInt32>.stride,1),options:.storageModeShared)!; return (vb,ib,0) }
        let vb = device.makeBuffer(bytes: vertices, length: vertices.count*MemoryLayout<Vertex>.stride, options: .storageModeShared)!; vb.label="VB_\(UUID())"
        let ib = device.makeBuffer(bytes: indices, length: indices.count*MemoryLayout<UInt32>.stride, options: .storageModeShared)!; ib.label="IB_\(UUID())"
        return (vb, ib, indices.count)
    }

    func createPlayingSurface() -> Mesh {
        let hL = Constants.tableLength/2.0; let hW = Constants.tableWidth/2.0; let y = Constants.playingSurfaceElevation
        // Correct Initializer
        let v = [
            Vertex(position:[-hL,y,-hW], normal:[0,1,0], texCoords:[0,0]),
            Vertex(position:[ hL,y,-hW], normal:[0,1,0], texCoords:[1,0]),
            Vertex(position:[ hL,y, hW], normal:[0,1,0], texCoords:[1,1]),
            Vertex(position:[-hL,y, hW], normal:[0,1,0], texCoords:[0,1])
        ]
        let idx:[UInt32]=[0,1,2,0,2,3]; let(vb,ib,ic)=createBuffers(vertices:v,indices:idx); let mat=Material(baseColor:Constants.feltColor,roughness:Constants.feltRoughness,metallic:Constants.feltMetallic,specular:Constants.feltSpecular,usesTexture:true); return Mesh(vertexBuffer:vb,indexBuffer:ib,indexCount:ic,primitiveType:.triangle,material:mat)
    }

    func createRailSegment(length: Float, width: Float, position: SIMD3<Float>, rotationY: Float = 0) -> Mesh {
        let hL = length/2.0; let hW = width/2.0; let yB = Constants.playingSurfaceElevation; let yT = Constants.tableHeight; let height = yT - yB
        // Correct Initializer
        let v = [
            Vertex(position:[-hL, height, -hW], normal:[0,1,0], texCoords:[0,0]), Vertex(position:[ hL, height, -hW], normal:[0,1,0], texCoords:[1,0]), Vertex(position:[ hL, height,  hW], normal:[0,1,0], texCoords:[1,1]), Vertex(position:[-hL, height,  hW], normal:[0,1,0], texCoords:[0,1]),
            Vertex(position:[-hL, 0, -hW], normal:[0,-1,0], texCoords:[0,0]), Vertex(position:[ hL, 0, -hW], normal:[0,-1,0], texCoords:[1,0]), Vertex(position:[ hL, 0,  hW], normal:[0,-1,0], texCoords:[1,1]), Vertex(position:[-hL, 0,  hW], normal:[0,-1,0], texCoords:[0,1]),
            Vertex(position:[-hL, height, hW], normal:[0,0,1], texCoords:[0,1]), Vertex(position:[ hL, height, hW], normal:[0,0,1], texCoords:[1,1]), Vertex(position:[ hL, 0, hW], normal:[0,0,1], texCoords:[1,0]), Vertex(position:[-hL, 0, hW], normal:[0,0,1], texCoords:[0,0]),
            Vertex(position:[ hL, height,-hW], normal:[0,0,-1], texCoords:[1,1]), Vertex(position:[-hL, height,-hW], normal:[0,0,-1], texCoords:[0,1]), Vertex(position:[-hL, 0,-hW], normal:[0,0,-1], texCoords:[0,0]), Vertex(position:[ hL, 0,-hW], normal:[0,0,-1], texCoords:[1,0]),
            Vertex(position:[ hL, height,-hW], normal:[1,0,0], texCoords:[0,1]), Vertex(position:[ hL, height, hW], normal:[1,0,0], texCoords:[1,1]), Vertex(position:[ hL, 0, hW], normal:[1,0,0], texCoords:[1,0]), Vertex(position:[ hL, 0,-hW], normal:[1,0,0], texCoords:[0,0]),
            Vertex(position:[-hL, height, hW], normal:[-1,0,0], texCoords:[0,1]), Vertex(position:[-hL, height,-hW], normal:[-1,0,0], texCoords:[1,1]), Vertex(position:[-hL, 0,-hW], normal:[-1,0,0], texCoords:[1,0]), Vertex(position:[-hL, 0, hW], normal:[-1,0,0], texCoords:[0,0]),
        ].map { Vertex(position: $0.position + [0, yB, 0], normal: $0.normal, texCoords: $0.texCoords) }

        let idx:[UInt32]=[0,1,2,0,2,3, 4,7,6,4,6,5, 8,9,10,8,10,11, 12,13,14,12,14,15, 16,17,18,16,18,19, 20,21,22,20,22,23]
        let rotM=matrix_float4x4_rotation(radians:rotationY,axis:[0,1,0]); let transM=matrix_float4x4_translation(position); let transform=transM*rotM
        var tVerts=v; for i in 0..<v.count {
            let p4=transform * SIMD4<Float>(v[i].position,1); tVerts[i].position=SIMD3<Float>(p4.x,p4.y,p4.z)/p4.w
            // Correct SIMD member access and explicit SIMD3 creation
            let u0 = SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z)
            let u1 = SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z)
            let u2 = SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
            let u3=simd_float3x3(u0, u1, u2)
            if abs(simd_determinant(u3))>Float.ulpOfOne{let nM=u3.inverse.transpose; tVerts[i].normal=normalize(nM*v[i].normal)}else{tVerts[i].normal=[0,1,0]}
        }
        let(vb,ib,ic)=createBuffers(vertices:tVerts,indices:idx); let mat=Material(baseColor:Constants.woodColor,roughness:Constants.woodRoughness,metallic:Constants.woodMetallic,specular:Constants.woodSpecular,usesTexture:true); return Mesh(vertexBuffer:vb,indexBuffer:ib,indexCount:ic,primitiveType:.triangle,material:mat)
    }

    // --- K55 Cushion Segment --- (Corrected Initializer)
    func createCushionSegment(length: Float, position: SIMD3<Float>, rotationY: Float = 0) -> Mesh {
        let halfL = length / 2.0
        let k55_SubH = Constants.k55_SubrailHeight; let k55_NoseH = Constants.k55_NoseHeight

        // Correct Initializer using SIMD2<Float>(x:y:)
        let profile: [SIMD2<Float>] = [
            SIMD2<Float>(x: -0.05, y: 0),                               // Back-bottom (clearly visible depth)
            SIMD2<Float>(x: 0.0, y: 0),                                 // Front-bottom (aligned vertically with nose)
            SIMD2<Float>(x: 0.0, y: Constants.k55_NoseHeight),          // Nose tip
            SIMD2<Float>(x: -0.03, y: Constants.k55_SubrailHeight),     // Angled top-back edge (distinct angle)
            SIMD2<Float>(x: -0.05, y: Constants.k55_SubrailHeight)      // Top-back edge (clearly visible)
        ]
        let profilePointCount = profile.count

        var finalVertices: [Vertex] = []
        var finalIndices: [UInt32] = []
        var currentIndex: UInt32 = 0
        let baseElevation = Constants.playingSurfaceElevation

        var faceNormals: [SIMD3<Float>] = []
        for i in 0..<profilePointCount { let p0=profile[i]; let p1=profile[(i+1)%profilePointCount]; let dx=p1.x-p0.x; let dy=p1.y-p0.y; let edge=SIMD3<Float>(0,dx,dy); faceNormals.append(normalize(cross(edge,[1,0,0]))) }

        for i in 0..<profilePointCount-1 {
            let faceNormal = faceNormals[i]
            let uv0 = Float(i)/Float(profilePointCount-1); let uv1 = Float(i+1)/Float(profilePointCount-1)
            let p0 = profile[i]; let p1 = profile[i+1]
            let v0L = SIMD3<Float>(-halfL, baseElevation + p0.y, p0.x)
            let v1L = SIMD3<Float>(-halfL, baseElevation + p1.y, p1.x)
            let v0R = SIMD3<Float>( halfL, baseElevation + p0.y, p0.x)
            let v1R = SIMD3<Float>( halfL, baseElevation + p1.y, p1.x)

            finalVertices.append(Vertex(position:v0L, normal:faceNormal, texCoords:[0,uv0]))
            finalVertices.append(Vertex(position:v0R, normal:faceNormal, texCoords:[1,uv0]))
            finalVertices.append(Vertex(position:v1R, normal:faceNormal, texCoords:[1,uv1]))
            finalVertices.append(Vertex(position:v1L, normal:faceNormal, texCoords:[0,uv1]))
            finalIndices.append(contentsOf:[currentIndex,currentIndex+1,currentIndex+2, currentIndex,currentIndex+2,currentIndex+3])
            currentIndex += 4
        }

        let addCap = { (normal:SIMD3<Float>, xPos:Float, reversed:Bool) in
            let startIdx = currentIndex
            for p in profile { let u=(p.y+0.025)/0.026; let v=p.x/k55_SubH; finalVertices.append(Vertex(position: [xPos, p.x+baseElevation, p.y], normal: normal, texCoords: [u,v])) }
            for i in 1..<profilePointCount-1 { if reversed { finalIndices.append(contentsOf: [startIdx, startIdx+UInt32(i+1), startIdx+UInt32(i)]) } else { finalIndices.append(contentsOf: [startIdx, startIdx+UInt32(i), startIdx+UInt32(i+1)]) } }
            currentIndex += UInt32(profilePointCount)
        }
        addCap([-1,0,0],-halfL,false); addCap([1,0,0],halfL,true)

        let rotM=matrix_float4x4_rotation(radians:rotationY,axis:[0,1,0]); let transM=matrix_float4x4_translation(position); let transform=transM*rotM
        var tVerts=finalVertices; for i in 0..<finalVertices.count {
            let p4=transform*SIMD4<Float>(finalVertices[i].position,1); tVerts[i].position=SIMD3<Float>(p4.x,p4.y,p4.z)/p4.w
            // Correct SIMD member access and explicit SIMD3 creation
            let u0 = SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z)
            let u1 = SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z)
            let u2 = SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
            let u3=simd_float3x3(u0, u1, u2)
            if abs(simd_determinant(u3))>Float.ulpOfOne{let nM=u3.inverse.transpose; tVerts[i].normal=normalize(nM*finalVertices[i].normal)}else{tVerts[i].normal=normalize(finalVertices[i].normal)}}

        let(vb,ib,ic)=createBuffers(vertices:tVerts,indices:finalIndices); let mat=Material(baseColor:Constants.feltColor,roughness:Constants.cushionRoughness,metallic:Constants.cushionMetallic,specular:Constants.cushionSpecular,usesTexture:false); return Mesh(vertexBuffer:vb,indexBuffer:ib,indexCount:ic,primitiveType:.triangle,material:mat)
    }

    func createPocketLiner(radius: Float, depth: Float, position: SIMD3<Float>, segments: Int = 16) -> Mesh {
        var v:[Vertex]=[]; var idx:[UInt32]=[]; let yT=Constants.playingSurfaceElevation; let yB=yT-depth
        for i in 0...segments{let a=Float(i)/Float(segments) * 2 * .pi;let x=cos(a) * radius;let z=sin(a)*radius;let n=normalize(SIMD3<Float>(x,0,z));v.append(Vertex(position:[x,yT,z], normal:n, texCoords:[Float(i)/Float(segments),0]))} // Correct init
        let topS:UInt32=0;let botS:UInt32=UInt32(v.count)
        for i in 0...segments{let a=Float(i)/Float(segments) * 2 * .pi;let x=cos(a) * radius;let z=sin(a)*radius;let n=normalize(SIMD3<Float>(x,0,z));v.append(Vertex(position:[x,yB,z], normal:n, texCoords:[Float(i)/Float(segments),1]))} // Correct init
        for i in 0..<segments{let i0=topS+UInt32(i);let i1=topS+UInt32(i+1);let i2=botS+UInt32(i);let i3=botS+UInt32(i+1);idx.append(contentsOf:[i0,i2,i1,i1,i2,i3])}
        let transform=matrix_float4x4_translation(position);var tVerts=v;for i in 0..<v.count{let p4=transform*SIMD4<Float>(v[i].position,1);tVerts[i].position=SIMD3<Float>(p4.x,p4.y,p4.z)/p4.w}
        let(vb,ib,ic)=createBuffers(vertices:tVerts,indices:idx);let mat=Material(baseColor:Constants.leatherColor,roughness:Constants.leatherRoughness,metallic:Constants.leatherMetallic,specular:Constants.leatherSpecular,usesTexture:false);return Mesh(vertexBuffer:vb,indexBuffer:ib,indexCount:ic,primitiveType:.triangle,material:mat)
    }

    func createMarker(position: SIMD3<Float>) -> Mesh {
        let sz=Constants.markerRadius*1.414;let y=Constants.tableHeight-0.005
        // Correct Init
        let v=[
            Vertex(position:[-sz/2,y,-sz/2], normal:[0,1,0], texCoords:[0,0]),
            Vertex(position:[ sz/2,y,-sz/2], normal:[0,1,0], texCoords:[1,0]),
            Vertex(position:[ sz/2,y, sz/2], normal:[0,1,0], texCoords:[1,1]),
            Vertex(position:[-sz/2,y, sz/2], normal:[0,1,0], texCoords:[0,1])
        ]
        let idx:[UInt32]=[0,1,2,0,2,3];let transform=matrix_float4x4_translation(position);var tVerts=v;for i in 0..<v.count{let p4=transform*SIMD4<Float>(v[i].position,1);tVerts[i].position=SIMD3<Float>(p4.x,p4.y,p4.z)/p4.w}
        let(vb,ib,ic)=createBuffers(vertices:tVerts,indices:idx);let mat=Material(baseColor:Constants.markerColor,roughness:Constants.markerRoughness,metallic:Constants.markerMetallic,specular:Constants.markerSpecular,usesTexture:false);return Mesh(vertexBuffer:vb,indexBuffer:ib,indexCount:ic,primitiveType:.triangle,material:mat)
    }

    // --- Build the Full Table --- (Using Revised Placement)
    func createFullTable() -> [Mesh] {
        var meshes: [Mesh] = []; meshes.append(createPlayingSurface())
        let hL=Constants.tableLength/2.0; let hW=Constants.tableWidth/2.0; let cornerCut=Constants.cornerPocketOpening/2.0; let sideCut=Constants.sidePocketOpening/2.0
        let railH=Constants.tableHeight-Constants.playingSurfaceElevation; let railW=Constants.railWidth
        let longSegLen=(Constants.tableLength-2*cornerCut-Constants.sidePocketOpening)/2.0; let shortSegLen=Constants.tableWidth-2*cornerCut
        let railPosZ=hW+railW / 2.0; let railPosX = hL + railW / 2.0; let seg1X = -(hL-cornerCut-longSegLen/2.0); let seg2X=(hL-cornerCut-longSegLen/2.0)
        meshes.append(createRailSegment(length:longSegLen,width:railW,position:[seg1X,0,railPosZ],rotationY:0)); meshes.append(createRailSegment(length:longSegLen,width:railW,position:[seg2X,0,railPosZ],rotationY:0))
        meshes.append(createRailSegment(length:longSegLen,width:railW,position:[seg1X,0,-railPosZ],rotationY:0)); meshes.append(createRailSegment(length:longSegLen,width:railW,position:[seg2X,0,-railPosZ],rotationY:0))
        meshes.append(createRailSegment(length:shortSegLen,width:railW,position:[railPosX,0,0],rotationY:.pi/2.0)); meshes.append(createRailSegment(length:shortSegLen,width:railW,position:[-railPosX,0,0],rotationY:.pi/2.0))
        let longCushLen=longSegLen; let shortCushLen=shortSegLen; let cushionPosZ=hW; let cushionPosX=hL
        meshes.append(createCushionSegment(length:longCushLen,position:[seg1X,0,cushionPosZ],rotationY:.pi)); meshes.append(createCushionSegment(length:longCushLen,position:[seg2X,0,cushionPosZ],rotationY:.pi))
        meshes.append(createCushionSegment(length:longCushLen,position:[seg1X,0,-cushionPosZ],rotationY:0)); meshes.append(createCushionSegment(length:longCushLen,position:[seg2X,0,-cushionPosZ],rotationY:0))
        meshes.append(createCushionSegment(length:shortCushLen,position:[cushionPosX,0,0],rotationY:-.pi/2.0)); meshes.append(createCushionSegment(length:shortCushLen,position:[-cushionPosX,0,0],rotationY:.pi/2.0))
        let cPR=Constants.cornerPocketOpening/2.0;let sPR=Constants.sidePocketOpening/2.0;let pPos=[[hL,0,hW],[-hL,0,hW],[hL,0,-hW],[-hL,0,-hW],[0,0,hW],[0,0,-hW]].map{SIMD3<Float>($0)};let pR=[cPR,cPR,cPR,cPR,sPR,sPR];for i in 0..<pPos.count{meshes.append(createPocketLiner(radius:pR[i],depth:Constants.pocketDepth,position:pPos[i]))}
        let mY=Constants.tableHeight-0.005;let mAreaL=Constants.tableLength-2*cornerCut;let mAreaW=Constants.tableWidth-2*cornerCut;let mSpaceL=mAreaL/4.0;let mSpaceW=mAreaW/2.0;let mRailCZ=hW+railW/2.0;let mRailCX=hL+railW/2.0;let mPos=[[-mSpaceL,mY,mRailCZ],[0,mY,mRailCZ],[mSpaceL,mY,mRailCZ],[-mSpaceL,mY,-mRailCZ],[0,mY,-mRailCZ],[mSpaceL,mY,-mRailCZ],[mRailCX,mY,0],[-mRailCX,mY,0]].map{SIMD3<Float>($0)};for p in mPos{meshes.append(createMarker(position:p))}
        print("Gen \(meshes.count) meshes.");return meshes
    }
}

// MARK: - Procedural Texture Generation (Simplified Solid Colors)
class TextureGenerator {
    let device: MTLDevice; init(device: MTLDevice){self.device=device}
    func generateFeltTexture(width:Int,height:Int)->MTLTexture?{let desc=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba8Unorm,width:width,height:height,mipmapped:false);desc.usage=[.shaderRead,.shaderWrite];guard let tex=device.makeTexture(descriptor:desc)else{print("Err felt");return nil};tex.label="FeltSimple";let bpp=4;let bpr=width*bpp;let bc=width*height*bpp;var data=[UInt8](repeating:0,count:bc);let R=UInt8(Constants.feltColor.x*255);let G=UInt8(Constants.feltColor.y*255);let B=UInt8(Constants.feltColor.z*255);for i in stride(from:0,to:bc,by:bpp){if i+3<bc{data[i]=R;data[i+1]=G;data[i+2]=B;data[i+3]=255}};tex.replace(region:MTLRegionMake2D(0,0,width,height),mipmapLevel:0,withBytes:data,bytesPerRow:bpr);print("Gen simple felt.");return tex}
    func generateWoodTexture(width:Int,height:Int)->MTLTexture?{let desc=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba8Unorm,width:width,height:height,mipmapped:false);desc.usage=[.shaderRead,.shaderWrite];guard let tex=device.makeTexture(descriptor:desc)else{print("Err wood");return nil};tex.label="WoodSimple";let bpp=4;let bpr=width*bpp;let bc=width*height*bpp;var data=[UInt8](repeating:0,count:bc);let R=UInt8(Constants.woodColor.x*255);let G=UInt8(Constants.woodColor.y*255);let B=UInt8(Constants.woodColor.z*255);for i in stride(from:0,to:bc,by:bpp){if i+3<bc{data[i]=R;data[i+1]=G;data[i+2]=B;data[i+3]=255}};tex.replace(region:MTLRegionMake2D(0,0,width,height),mipmapLevel:0,withBytes:data,bytesPerRow:bpr);print("Gen simple wood.");return tex}
}


// MARK: - Metal Shaders (MSL) (Unchanged - Not minified)
let shaderSource = """
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct VertexIn{float3 position[[attribute(0)]];float3 normal[[attribute(1)]];float2 texCoords[[attribute(2)]];};
struct VertexOut{float4 position[[position]];float3 worldPosition;float3 worldNormal;float2 texCoords;float4 lightSpacePosition;};
struct Uniforms{float4x4 modelMatrix;float4x4 viewMatrix;float4x4 projectionMatrix;float3x3 normalMatrix;float3 cameraWorldPosition;float3 lightWorldPosition;float4x4 lightViewMatrix;float4x4 lightProjectionMatrix;};
struct Material{float3 baseColor;float roughness;float metallic;float specular;float ambientOcclusion;bool usesTexture;};
float D_GGX(float NoH,float roughness){float a=roughness*roughness;float a2=a*a;float NoH2=NoH*NoH;float den=(NoH2*(a2-1.0)+1.0);return a2/max(M_PI_F*den*den,1e-6f);}
float G_SchlickGGX_Sub(float NoV_or_NoL,float roughness){float r=(roughness+1.0);float k=(r*r)/8.0;return NoV_or_NoL/max(NoV_or_NoL*(1.0-k)+k,1e-6f);}
float G_Smith(float NoV,float NoL,float roughness){float ggxV=G_SchlickGGX_Sub(NoV,roughness);float ggxL=G_SchlickGGX_Sub(NoL,roughness);return ggxV*ggxL;}
float3 F_Schlick(float cosTheta,float3 F0){float Fc=pow(1.0-saturate(cosTheta),5.0);return F0+(1.0-F0)*Fc;}
vertex VertexOut vertex_main(VertexIn in[[stage_in]],constant Uniforms &u[[buffer(1)]]){VertexOut o;float4 wp4=u.modelMatrix*float4(in.position,1.0);o.worldPosition=wp4.xyz/wp4.w;o.position=u.projectionMatrix*u.viewMatrix*wp4;o.worldNormal=normalize(u.normalMatrix*in.normal);o.texCoords=in.texCoords;o.lightSpacePosition=u.lightProjectionMatrix*u.lightViewMatrix*wp4;return o;}
fragment float4 fragment_main(VertexOut in[[stage_in]],constant Uniforms &u[[buffer(1)]],constant Material &mat[[buffer(2)]],texture2d<float> tex[[texture(0)]],depth2d<float> shadowMap[[texture(1)]],sampler linSamp[[sampler(0)]],sampler shadowSamp[[sampler(1)]]){
float3 N=normalize(in.worldNormal);float3 V=normalize(u.cameraWorldPosition-in.worldPosition);float3 L=normalize(u.lightWorldPosition-in.worldPosition);float3 H=normalize(V+L);
float NoV=saturate(dot(N,V));float NoL=saturate(dot(N,L));float NoH=saturate(dot(N,H));float VoH=saturate(dot(V,H));
float3 alb=mat.baseColor;if(mat.usesTexture){alb=tex.sample(linSamp,in.texCoords).rgb*mat.baseColor;}
float rough=max(mat.roughness,0.04);float met=mat.metallic;float ao=mat.ambientOcclusion;float3 F0=mix(float3(0.04),alb,met);F0=mix(F0,F0*mat.specular,1.0-met);
float NDF=D_GGX(NoH,rough);float G=G_Smith(NoV,NoL,rough);float3 F=F_Schlick(VoH,F0);
float3 num=NDF*G*F;float den=4.0*NoV*NoL+1e-4f;float3 spec=num/den;
float3 kS=F;float3 kD=(float3(1.0)-kS)*(1.0-met);float3 diff=kD*alb/M_PI_F;
float shadow=1.0;float3 projC=in.lightSpacePosition.xyz/in.lightSpacePosition.w;float2 shadowUV=projC.xy*0.5+0.5;shadowUV.y=1.0-shadowUV.y;
if(shadowUV.x>=0.0&&shadowUV.x<=1.0&&shadowUV.y>=0.0&&shadowUV.y<=1.0&&projC.z<=1.0){
float curD=projC.z;float bias=max(0.005*tan(acos(NoL)),0.0005);float shadowAcc=0.0;float texelS=1.0/float(shadowMap.get_width());
for(int x=-1;x<=0;++x){for(int y=-1;y<=0;++y){shadowAcc+=shadowMap.sample_compare(shadowSamp,shadowUV+float2(x,y)*texelS,curD-bias);}}shadow=shadowAcc/4.0;}
float3 lightCol=float3(1.0,1.0,0.95)*3.5;float dist=length(u.lightWorldPosition-in.worldPosition);float atten=1.0/(dist*dist+1.0);
float3 Lo=(diff+spec)*NoL*lightCol*atten*shadow;float3 amb=float3(0.05,0.05,0.07)*alb*ao;float3 finalC=amb+Lo;
finalC=finalC/(finalC+float3(1.0));finalC=pow(finalC,float3(1.0/2.2));return float4(finalC,1.0);}
vertex VertexOut shadow_vertex_main(VertexIn in[[stage_in]],constant Uniforms &u[[buffer(1)]]){VertexOut o;float4 wp4=u.modelMatrix*float4(in.position,1.0);o.position=u.lightProjectionMatrix*u.lightViewMatrix*wp4;o.worldPosition=0;o.worldNormal=0;o.texCoords=0;o.lightSpacePosition=0;return o;}
"""

// MARK: - Renderer Class (Unchanged Core Logic, Not Minified)
class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice; let commandQueue: MTLCommandQueue
    var renderPipelineState: MTLRenderPipelineState!; var shadowPipelineState: MTLRenderPipelineState!
    var depthStencilState: MTLDepthStencilState!; var shadowDepthStencilState: MTLDepthStencilState!
    var mainSamplerState: MTLSamplerState!; var shadowSamplerState: MTLSamplerState!
    var meshes: [Mesh] = []; var uniforms: Uniforms; var uniformBuffer: MTLBuffer!; var materialUniformBuffer: MTLBuffer!
    var time: Float = 0; let geometryGenerator: GeometryGenerator; let textureGenerator: TextureGenerator
    var feltTexture: MTLTexture?; var woodTexture: MTLTexture?; var shadowDepthTexture: MTLTexture?
    var cameraPosition = SIMD3<Float>(0, 2.5, -4.5); var cameraTarget = SIMD3<Float>(0, Constants.playingSurfaceElevation, 0); var lightPosition = SIMD3<Float>(1.5, 4.5, -2.5)

    init?(metalKitView: MTKView) {
        guard let dev = metalKitView.device, let q = dev.makeCommandQueue() else { print("Rend Err: No device/queue"); return nil }
        self.device = dev; self.commandQueue = q
        self.geometryGenerator = GeometryGenerator(device: dev); self.textureGenerator = TextureGenerator(device: dev)
        self.uniforms = Uniforms(modelMatrix:matrix_identity_float4x4, viewMatrix:matrix_identity_float4x4, projectionMatrix:matrix_identity_float4x4, normalMatrix:matrix_identity_float3x3, cameraWorldPosition:self.cameraPosition, lightWorldPosition:self.lightPosition, lightViewMatrix:matrix_identity_float4x4, lightProjectionMatrix:matrix_identity_float4x4)
        super.init()
        metalKitView.delegate = self; metalKitView.depthStencilPixelFormat = .depth32Float; metalKitView.colorPixelFormat = .bgra8Unorm_srgb; metalKitView.clearColor = MTLClearColor(red:0.05, green:0.05, blue:0.08, alpha:1.0); metalKitView.sampleCount = 1
        do { try setupPipelineStates(pixelFormat: metalKitView.colorPixelFormat, depthFormat: metalKitView.depthStencilPixelFormat); setupDepthStencilStates(); setupSamplers(); setupShadowMapTexture(); createUniformBuffers(); loadAssets() } catch { print("Renderer setup Error: \(error)"); return nil }
    }

    func createUniformBuffers() { uniformBuffer = device.makeBuffer(length:MemoryLayout<Uniforms>.stride, options:.storageModeShared); uniformBuffer.label="Uniforms"; materialUniformBuffer = device.makeBuffer(length:MemoryLayout<Material>.stride, options:.storageModeShared); materialUniformBuffer.label="Material" }

    func setupPipelineStates(pixelFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {
        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else { throw NSError(domain:"Pipe", code:1, userInfo:[NSLocalizedDescriptionKey:"Lib fail"]) }
        let vFunc = library.makeFunction(name: "vertex_main"); let fFunc = library.makeFunction(name: "fragment_main")
        let pDesc = MTLRenderPipelineDescriptor(); pDesc.label="PBR"; pDesc.vertexFunction=vFunc; pDesc.fragmentFunction=fFunc
        let vDesc = MTLVertexDescriptor(); vDesc.attributes[0].format = .float3; vDesc.attributes[0].offset=0; vDesc.attributes[0].bufferIndex=0; vDesc.attributes[1].format = .float3; vDesc.attributes[1].offset=MemoryLayout<SIMD3<Float>>.stride; vDesc.attributes[1].bufferIndex=0; vDesc.attributes[2].format = .float2; vDesc.attributes[2].offset=MemoryLayout<SIMD3<Float>>.stride*2; vDesc.attributes[2].bufferIndex=0; vDesc.layouts[0].stride=MemoryLayout<Vertex>.stride; pDesc.vertexDescriptor=vDesc
        pDesc.colorAttachments[0].pixelFormat=pixelFormat; pDesc.depthAttachmentPixelFormat=depthFormat; renderPipelineState = try device.makeRenderPipelineState(descriptor: pDesc)
        let sVFunc = library.makeFunction(name: "shadow_vertex_main"); let sPDesc = MTLRenderPipelineDescriptor(); sPDesc.label="Shadow"; sPDesc.vertexFunction=sVFunc; sPDesc.fragmentFunction=nil; sPDesc.vertexDescriptor=vDesc; sPDesc.depthAttachmentPixelFormat = .depth32Float; sPDesc.colorAttachments[0].pixelFormat = .invalid; shadowPipelineState = try device.makeRenderPipelineState(descriptor: sPDesc)
    }

    func setupDepthStencilStates() { let dDesc=MTLDepthStencilDescriptor();dDesc.depthCompareFunction = .lessEqual;dDesc.isDepthWriteEnabled=true;depthStencilState=device.makeDepthStencilState(descriptor:dDesc)!;let sDesc=MTLDepthStencilDescriptor();sDesc.depthCompareFunction = .lessEqual;sDesc.isDepthWriteEnabled=true;shadowDepthStencilState=device.makeDepthStencilState(descriptor:sDesc)! }
    func setupSamplers() { let mS=MTLSamplerDescriptor();mS.minFilter = .linear;mS.magFilter = .linear;mS.mipFilter = .linear;mS.sAddressMode = .repeat;mS.tAddressMode = .repeat;mainSamplerState=device.makeSamplerState(descriptor:mS)!;let sS=MTLSamplerDescriptor();sS.minFilter = .linear;sS.magFilter = .linear;sS.compareFunction = .lessEqual;shadowSamplerState=device.makeSamplerState(descriptor:sS)! }
    func setupShadowMapTexture() { let desc=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.depth32Float,width:Constants.shadowMapSize,height:Constants.shadowMapSize,mipmapped:false); desc.usage=[.renderTarget,.shaderRead]; desc.storageMode = .private; guard let tex=device.makeTexture(descriptor:desc) else { fatalError("ShadowTex fail") }; shadowDepthTexture=tex; shadowDepthTexture?.label="Shadow Depth" }
    func loadAssets() { print("Load assets...");meshes=geometryGenerator.createFullTable();print("Gen tex (SMALL)...");feltTexture=textureGenerator.generateFeltTexture(width:64,height:64);woodTexture=textureGenerator.generateWoodTexture(width:128,height:64);for i in 0..<meshes.count{if meshes[i].material.usesTexture{if meshes[i].material.baseColor==Constants.feltColor{meshes[i].texture=feltTexture}else if meshes[i].material.baseColor==Constants.woodColor{meshes[i].texture=woodTexture}}};print("Assets loaded.") }

    func rotateCamera(translation:CGSize) { let s:Float=0.005;let aX=Float(translation.width)*s;let aY=Float(translation.height)*s*0.5;let curPR=cameraPosition-cameraTarget;let cA=cos(aX);let sA=sin(aX);let rX=curPR.x*cA-curPR.z*sA;let rZ=curPR.x*sA+curPR.z*cA;var nPR=SIMD3<Float>(rX,curPR.y,rZ);let lookH=normalize(SIMD2<Float>(nPR.x,nPR.z));let rAxis=SIMD3<Float>(lookH.y,0,-lookH.x);let vRot=matrix_float4x4_rotation(radians:-aY,axis:rAxis);let rP4=vRot*SIMD4<Float>(nPR,0);nPR=SIMD3<Float>(rP4.x,rP4.y,rP4.z);cameraPosition=nPR+cameraTarget;let minH:Float=0.2;let maxH:Float=6.0;let curH=cameraPosition.y-cameraTarget.y;let hDistSq=max(Float.ulpOfOne,nPR.x*nPR.x+nPR.z*nPR.z);let origHDistSq=max(Float.ulpOfOne,curPR.x*curPR.x+curPR.z*curPR.z);let scale=sqrt(origHDistSq/hDistSq);if curH<minH{let cY=cameraTarget.y+minH;cameraPosition=SIMD3<Float>(nPR.x*scale,cY,nPR.z*scale)+cameraTarget}else if curH>maxH{let cY=cameraTarget.y+maxH;cameraPosition=SIMD3<Float>(nPR.x*scale,cY,nPR.z*scale)+cameraTarget}}

    func updateUniforms(view:MTKView){ time+=1.0/Float(max(1,view.preferredFramesPerSecond));uniforms.viewMatrix=matrix_look_at_right_hand(eye:cameraPosition,center:cameraTarget,up:[0,1,0]);let aspect=Float(view.drawableSize.width/max(1.0,view.drawableSize.height));uniforms.projectionMatrix=matrix_perspective_right_hand(fovyRadians:Float(65).degreesToRadians,aspect:aspect,nearZ:0.1,farZ:100.0);uniforms.lightWorldPosition=lightPosition;uniforms.cameraWorldPosition=cameraPosition;uniforms.lightViewMatrix=matrix_look_at_right_hand(eye:uniforms.lightWorldPosition,center:cameraTarget,up:[0,1,0]);let sOS:Float=3.5;uniforms.lightProjectionMatrix=matrix_ortho_right_hand(left:-sOS,right:sOS,bottom:-sOS,top:sOS,nearZ:0.1,farZ:15.0);uniforms.modelMatrix=matrix_identity_float4x4;uniforms.normalMatrix=matrix_identity_float3x3;uniformBuffer.contents().copyMemory(from:&uniforms,byteCount:MemoryLayout<Uniforms>.stride) }
    func createShadowRenderPassDescriptor()->MTLRenderPassDescriptor?{ guard let tex=shadowDepthTexture else{return nil};let d=MTLRenderPassDescriptor();d.depthAttachment.texture=tex;d.depthAttachment.loadAction = .clear;d.depthAttachment.storeAction = .store;d.depthAttachment.clearDepth=1.0;d.colorAttachments[0].loadAction = .dontCare;d.colorAttachments[0].storeAction = .dontCare;d.colorAttachments[0].texture=nil;return d}
    func mtkView(_ view:MTKView,drawableSizeWillChange size:CGSize){updateUniforms(view:view)}

    func draw(in view: MTKView) {
        guard let cmdBuf=commandQueue.makeCommandBuffer() else {return}; cmdBuf.label="Frame"; updateUniforms(view:view)
        // Shadow Pass
        if let spd=createShadowRenderPassDescriptor(), let se=cmdBuf.makeRenderCommandEncoder(descriptor:spd){ se.label="Shadow"; se.setRenderPipelineState(shadowPipelineState); se.setDepthStencilState(shadowDepthStencilState); se.setCullMode(.front); se.setDepthBias(0.01,slopeScale:1.5,clamp:0.02); se.setVertexBuffer(uniformBuffer,offset:0,index:1); for m in meshes{ se.setVertexBuffer(m.vertexBuffer,offset:0,index:0); se.drawIndexedPrimitives(type:m.primitiveType,indexCount:m.indexCount,indexType:.uint32,indexBuffer:m.indexBuffer,indexBufferOffset:0)}; se.endEncoding() }
        // Main Pass
        if let rpd=view.currentRenderPassDescriptor, let re=cmdBuf.makeRenderCommandEncoder(descriptor:rpd){ re.label="Main"; re.setRenderPipelineState(renderPipelineState); re.setDepthStencilState(depthStencilState); re.setCullMode(.back); re.setVertexBuffer(uniformBuffer,offset:0,index:1); re.setFragmentBuffer(uniformBuffer,offset:0,index:1); re.setFragmentSamplerState(mainSamplerState,index:0); re.setFragmentSamplerState(shadowSamplerState,index:1); re.setFragmentTexture(shadowDepthTexture,index:1); for m in meshes{ re.setVertexBuffer(m.vertexBuffer,offset:0,index:0); var mat=m.material; materialUniformBuffer.contents().copyMemory(from:&mat,byteCount:MemoryLayout<Material>.stride); re.setFragmentBuffer(materialUniformBuffer,offset:0,index:2); re.setFragmentTexture(m.texture,index:0); re.drawIndexedPrimitives(type:m.primitiveType,indexCount:m.indexCount,indexType:.uint32,indexBuffer:m.indexBuffer,indexBufferOffset:0)}; re.endEncoding(); if let drw=view.currentDrawable{cmdBuf.present(drw)}}
        cmdBuf.commit()
    }
    func configure(view:MTKView){view.delegate=self;view.device=self.device;view.depthStencilPixelFormat = .depth32Float;view.clearColor=MTLClearColor(red:0.05,green:0.05,blue:0.08,alpha:1.0);view.colorPixelFormat = .bgra8Unorm_srgb;print("Rend config.")}
}

// MARK: - Helper Extensions
extension Float{var degreesToRadians:Float{return self * .pi/180};var radiansToDegrees:Float{return self * 180 / .pi}}
extension SIMD3 where Scalar==Float{init(_ v:[Float]){self.init(v[0],v[1],v[2])}} // Short init
extension SIMD2 where Scalar==Float{init(_ v:[Float]){self.init(v[0],v[1])}} // Short init

// MARK: - Coordinator Class
class MetalBilliardTableViewCoordinator{var renderer:Renderer?}

// MARK: - SwiftUI UIViewRepresentable
struct MetalBilliardTableViewRepresentable_Refined: UIViewRepresentable {
    @Binding var coordinator: Coordinator; typealias Coordinator = MetalBilliardTableViewCoordinator
    func makeCoordinator()->Coordinator{return MetalBilliardTableViewCoordinator()}
    func makeUIView(context:Context)->MTKView{let v=MTKView();v.enableSetNeedsDisplay=false;v.isPaused=false;guard let d=MTLCreateSystemDefaultDevice()else{fatalError("Metal fail")};v.device=d;if coordinator.renderer==nil{guard let r=Renderer(metalKitView:v)else{fatalError("Rend fail")};self.coordinator.renderer=r;print("Rend init.")}else{coordinator.renderer?.configure(view:v);print("Rend reconf.")};return v}
    func updateUIView(_ uiView:MTKView,context:Context){}
    static func dismantleUIView(_ uiView:MTKView,coordinator:Coordinator){print("Dismantle...")}
}

// MARK: - SwiftUI Content View
struct ContentView_Refined: View {
    @State private var coordinator = MetalBilliardTableViewCoordinator()
    @State private var dragOffset: CGSize = .zero; @GestureState private var isDragging = false
    var body: some View {
        MetalBilliardTableViewRepresentable_Refined(coordinator: $coordinator)
            .gesture(DragGesture(minimumDistance:0).updating($isDragging){_,s,_ in s=true}.onChanged{v in let d=CGSize(width:v.translation.width-dragOffset.width,height:v.translation.height-dragOffset.height);coordinator.renderer?.rotateCamera(translation:d);dragOffset=v.translation}.onEnded{_ in dragOffset = .zero})
            .edgesIgnoringSafeArea(.all)
    }
}

// MARK: - App Entry Point
@main
struct BilliardTableApp: App {
    var body: some Scene { WindowGroup { ContentView_Refined() } }
}
