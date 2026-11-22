# 2. Simulation, data and labels

<!-- toc -->

## Contents

- [1. MuJoCo](#1-mujoco)
  - [Big picture](#big-picture)
  - [In the weeds](#in-the-weeds)
  - [Gotchas](#gotchas)
  - [Try it (15 min)](#try-it-15-min)
  - [Read](#read)
- [2. Procedural generation and domain randomization](#2-procedural-generation-and-domain-randomization)
  - [Big picture](#big-picture-1)
  - [In the weeds](#in-the-weeds-1)
  - [Gotchas](#gotchas-1)
  - [Try it (15 min)](#try-it-15-min-1)
  - [Read](#read-1)
- [3. The sim-to-real gap](#3-the-sim-to-real-gap)
  - [Big picture](#big-picture-2)
  - [In the weeds](#in-the-weeds-2)
  - [Gotchas](#gotchas-2)
  - [Try it (15 min)](#try-it-15-min-2)
  - [Read](#read-2)
- [4. Datasets and labels](#4-datasets-and-labels)
  - [Big picture](#big-picture-3)
  - [In the weeds](#in-the-weeds-3)
  - [Gotchas](#gotchas-3)
  - [Try it (20 min)](#try-it-20-min)
  - [Read](#read-3)
- [5. Camera calibration and the ground-plane homography](#5-camera-calibration-and-the-ground-plane-homography)
  - [Big picture](#big-picture-4)
  - [In the weeds](#in-the-weeds-4)
  - [Gotchas](#gotchas-4)
  - [Try it (20 min)](#try-it-20-min-1)
  - [Read](#read-4)
- [6. Classical floor segmentation baseline](#6-classical-floor-segmentation-baseline)
  - [Big picture](#big-picture-5)
  - [In the weeds](#in-the-weeds-5)
  - [Gotchas](#gotchas-5)
  - [Try it (15 min)](#try-it-15-min-3)
  - [Read](#read-5)

<!-- tocstop -->

Versions checked on 2026-11-17. MuJoCo on PyPI is **4.0.1** (released Nov 11,
2026; 4.0.0 came out Nov 6). The code in sections 1, 2, 5 and 6 was run on an
Apple M1 with 8 GB using `mujoco==4.0.1`, Python 3.12 and
`opencv-python-headless`. The NYUv2 shapes in section 4 were read from the real
file.

______________________________________________________________________

## 1. MuJoCo

**Why it matters for Mini FSD.** MuJoCo is the world: it moves the robot,
renders the camera, and gives free depth and segmentation labels. You need it
working in Week 9 (sim + teleop) and you lean on its renderer all through Week
10 (data engine).

### Big picture

You describe a scene in an XML file called MJCF. MuJoCo compiles it into an
`MjModel`, which holds everything that does not change while the sim runs:
shapes, masses, joint limits, camera settings, textures. The changing state
lives in `MjData`: joint positions `qpos`, velocities `qvel`, controls `ctrl`,
contacts, and the world poses of every body.

Two functions do most of the work:

- `mj_step(m, d)` advances time by one timestep. Physics, contacts, actuators,
  integration.
- `mj_forward(m, d)` computes everything that follows from the current state
  (body poses, camera poses, contacts) but does not advance time. Call it after
  you set `qpos` by hand, before you render.

That second point matters for the data engine. To render a training frame you do
not need to simulate a robot driving there. Place the camera body, call
`mj_forward`, render.

### In the weeds

**MJCF structure.** The pieces you will use:

- `<option timestep="0.002"/>`: physics step in seconds. 0.002 is the default.
  Smaller is more stable but slower.
- `<asset>`: `<texture>` (built-in checker/gradient or a PNG file) and
  `<material>` (a texture plus color, `texrepeat`, `reflectance`).
- `<worldbody>`: the root of a tree of `<body>` elements. Each body has a pose
  relative to its parent and can hold:
  - `<geom>`: a shape (plane, box, sphere, capsule, cylinder, mesh). Geoms
    collide and are drawn.
  - `<joint>`: how this body moves relative to its parent. `hinge` (1 rotation),
    `slide` (1 translation), `ball`, `free` (6 DoF; use `<freejoint/>` for the
    robot base).
  - `<camera>` and `<light>`.
  - A body with `mocap="true"` is moved by you via `d.mocap_pos` and
    `d.mocap_quat`. Use it for scripted moving obstacles.
- `<actuator>`: what `d.ctrl` drives.
  - `motor`: force or torque = `gear * ctrl`.
  - `position`: a P controller toward `ctrl`, gain `kp`.
  - `velocity`: a velocity servo toward `ctrl`, gain `kv`. This is what you want
    on each wheel of a diff-drive robot.
- `<default>` classes let you set shared attributes once.
- `<visual><global offwidth="640" offheight="480"/></visual>` sets the offscreen
  framebuffer. The default is 640×480. Rendering larger fails.

A diff-drive robot that drives straight at the commanded speed on the M1 test:

```xml
<mujoco model="hall">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".3 .3 .3" rgb2=".5 .5 .5" width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor_mat"/>
    <geom name="box" type="box" pos="2 0 0.25" size="0.25 0.25 0.25" rgba="0.8 0.2 0.2 1"/>
    <body name="base" pos="0 0 0.1">
      <freejoint/>
      <geom type="cylinder" size="0.15 0.04" mass="3"/>
      <camera name="front" pos="0.15 0 0.2" xyaxes="0 -1 0 0 0 1" fovy="60"/>
      <body name="wl" pos="0 0.17 0" euler="-90 0 0">
        <joint name="left" type="hinge" axis="0 0 1"/>
        <geom type="cylinder" size="0.1 0.01" friction="1.5"/>
      </body>
      <body name="wr" pos="0 -0.17 0" euler="-90 0 0">
        <joint name="right" type="hinge" axis="0 0 1"/>
        <geom type="cylinder" size="0.1 0.01" friction="1.5"/>
      </body>
      <!-- casters sit 5 mm above the wheel bottoms; see Gotchas -->
      <geom type="sphere" size="0.03" pos="0.12 0 -0.065" friction="0.01 0.005 0.0001"/>
      <geom type="sphere" size="0.03" pos="-0.12 0 -0.065" friction="0.01 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vl" joint="left" kv="2"/>
    <velocity name="vr" joint="right" kv="2"/>
  </actuator>
</mujoco>
```

`cmd_vel` to wheel speeds, with wheel radius `r` and track width `L`:

```python
wl = (v - w * L / 2) / r
wr = (v + w * L / 2) / r
d.ctrl[:] = [wl, wr]
```

**Python basics.**

```python
import mujoco

m = mujoco.MjModel.from_xml_path("hall.xml")
d = mujoco.MjData(m)
d.ctrl[:] = [5.0, 5.0]
for _ in range(500):  # 1 s at 0.002
    mujoco.mj_step(m, d)
print(
    d.body("base").xpos, d.body("base").xquat
)  # world position, quaternion (w, x, y, z)
```

Named access (`d.body("base")`, `m.geom("floor").id`, `d.joint("left").qvel`)
saves you from index bookkeeping.

**Offscreen rendering with `mujoco.Renderer`.** One renderer, three modes:

```python
r = mujoco.Renderer(m, height=240, width=320)

r.update_scene(d, camera="front")
rgb = r.render()  # (240, 320, 3) uint8

r.enable_depth_rendering()
r.update_scene(d, camera="front")
depth = r.render()  # (240, 320) float32, meters
r.disable_depth_rendering()

r.enable_segmentation_rendering()
r.update_scene(d, camera="front")
seg = r.render()  # (240, 320, 2) int32: [object id, object type]
r.disable_segmentation_rendering()

floor = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (
    seg[..., 0] == m.geom("floor").id
)
```

Facts checked on the M1:

- Depth is distance along the camera's viewing axis (z-depth), in meters, not
  distance along the ray. A box face placed at 1, 3, 8 and 15 m read back within
  0.04 mm.
- Pixels that hit nothing get the far clip value
  (`m.vis.map.zfar * m.stat.extent`; 171.7 m in the test scene). Mask those out
  before saving.
- Segmentation background is `-1` in both channels. Channel 1 is the object type
  (5 = geom). To get one id per obstacle when an obstacle has several geoms, map
  geom to body with `m.geom_bodyid`.
- A 320×240 RGB render took about 2.6 ms. 1,000 physics steps of the small robot
  took about 9 ms.

**Camera axes.** A MuJoCo camera looks along its own **-z** axis, with **+y up**
and **+x right**. The ROS optical frame (REP 103) looks along **+z**, with **+y
down** and **+x right**. So the optical frame is the MuJoCo camera frame rotated
180° about x: flip y and z.

```python
R_mjcam_to_optical = np.diag([1, -1, -1])
```

In TF, publish `camera_link` (x forward, z up, the ROS body convention) and a
child `camera_optical_frame` with the usual rotation (roll -90°, yaw -90°).
Intrinsics for a MuJoCo camera with vertical field of view `fovy`:
`fy = (H/2) / tan(fovy/2)`, `fx = fy` for square pixels, `cx = W/2`, `cy = H/2`.

In MJCF, `xyaxes="x1 x2 x3 y1 y2 y3"` gives the camera's right (x) and up (y)
directions in the parent frame. `xyaxes="0 -1 0 0 0 1"` on the robot body means
right = robot's -y, up = +z, so the camera looks along robot +x.

**Contacts.** Two geoms can collide if
`(contype1 & conaffinity2) || (contype2 & conaffinity1)`. Both default to 1. Set
`contype="0" conaffinity="0"` for visual-only geoms like door signs. After a
step, `d.ncon` and `d.contact[i].geom1/geom2/dist` list the active contacts.
That is how your eval harness counts collisions: any contact between a robot
geom and an obstacle geom.

**What changed recently.** In 4.0, `pip install mujoco` became a metapackage
that pulls `mujoco-core` plus `mujoco-studio` (a new viewer and a Filament
renderer); `mujoco-core[rendering]` is the lean option. 4.0 dropped Python 3.10
and added "speculative contacts" against fast objects tunneling through thin
walls. MJX and MuJoCo Warp are for GPUs (Warp targets NVIDIA), so skip them on
the M1.

### Gotchas

- **Pin `mujoco==4.0.1`, not 4.0.0.** 4.0.0 had a sparse Cholesky regression
  that could give wrong accelerations. Fixed in 4.0.1.
- **The interactive viewer needs `mjpython` on macOS.**
  `mujoco.viewer.launch_passive` must run as `mjpython script.py`, because macOS
  requires rendering windows to live on the main thread. `mjpython` ships with
  the pip package.
- **Offscreen rendering is different.** On macOS the `Renderer` uses a CGL
  context by default (`MUJOCO_GL` can be `cgl` or `glfw`). In a test on the M1
  it rendered fine from a worker thread. Keep each `Renderer` in the thread that
  created it, since a GL context is current on one thread at a time. In an rclpy
  node, do the rendering in one callback group.
- **The warning
  `ARB_clip_control unavailable ... depth accuracy will be limited`** appears on
  macOS because Apple's OpenGL stops at 4.1. It is harmless at indoor ranges
  (see the depth check above).
- **Wheels slip if the casters touch the floor at the same height as the
  wheels.** With all four contacts at the same height, the wheels spun at 0.5
  m/s surface speed and the robot moved at 0.1 m/s. Raising the casters 5 mm
  fixed it (0.50 m/s, and turning rate matched the formula). Always check that
  commanded and measured speed agree.
- **8 GB RAM.** Write frames to disk as you go. If you parallelize generation,
  use 2–3 processes, each with its own `MjModel` and `Renderer`.

### Try it (15 min)

Save the XML above, drive the robot forward for 2 s, render RGB, depth and
segmentation from `front`, and save them as PNGs (colorize depth with
`cv2.applyColorMap`). Then check: does `(seg[...,0] == m.geom("box").id)` cover
the red box? Does the depth at the box center match the distance you expect?

### Read

- MJCF XML reference:
  <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- Python bindings (viewer, `mjpython`, rendering):
  <https://mujoco.readthedocs.io/en/stable/python.html>
- Modeling guide (contacts, solver, units):
  <https://mujoco.readthedocs.io/en/stable/modeling.html>
- Changelog (check before upgrading):
  <https://mujoco.readthedocs.io/en/stable/changelog.html>

______________________________________________________________________

## 2. Procedural generation and domain randomization

**Why it matters for Mini FSD.** The data engine is the core of the CV half. In
Week 10 you generate about 10k labeled frames from random buildings, and the E5
ablation (sim-only vs. sim + DR vs. sim + DR + real) depends on being able to
switch DR on and off.

### Big picture

Procedural generation means a program builds the building from a random seed:
hallway lengths, room layout, door positions, sign numbers, furniture. Same
seed, same building. That gives you reproducible data and a clean way to split
train and test.

Domain randomization (DR) means you vary everything the model should *not*
depend on: wall and floor textures, light color and position, camera height and
pitch, distractor objects, blur, noise. If the network sees enough variety in
sim, a real hallway looks like one more variation.

### In the weeds

**Seeding.** One seed per building. Derive per-frame randomness from it so a
frame can be regenerated:

```python
building_rng = np.random.default_rng(seed)
frame_rng = np.random.default_rng([seed, frame_idx])
```

Store the seed and every sampled parameter in a per-frame JSON. It helps
debugging and lets you slice results later ("does IoU drop at low camera
heights?").

**Build the scene in Python with `MjSpec`** instead of string-templating XML:

```python
spec = mujoco.MjSpec()
spec.option.timestep = 0.002
spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
spec.add_texture(
    name="floor_tex",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=rng.uniform(0, 1, 3),
    rgb2=rng.uniform(0, 1, 3),
    width=256,
    height=256,
)
mat = spec.add_material(name="floor_mat", texrepeat=[6, 6])
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
w = spec.worldbody
w.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
w.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10, 10, 0.1],
    material="floor_mat",
)
for i in range(3):
    w.add_geom(
        name=f"box{i}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[rng.uniform(1, 4), rng.uniform(-1, 1), 0.2],
        size=[0.2] * 3,
        rgba=[*rng.uniform(0, 1, 3), 1],
    )
w.add_camera(
    name="cam", pos=[0, 0, 0.5], xyaxes=[0, -1, 0, 0.34, 0, 0.94], fovy=60
)
m = spec.compile()
print(spec.to_xml()[:200])  # save the XML next to the frames for debugging
```

**Two levels of randomization.**

- Per building (needs a recompile): layout, which texture image goes on which
  wall, furniture meshes, door sign text. For signs, draw the number into a PNG
  with Pillow and load it as a texture.
- Per frame (no recompile): edit model arrays, call `mj_forward`, render. Tested
  on the M1, these change the next render: `m.light_diffuse`, `m.light_pos`,
  `m.geom_rgba`, `m.cam_fovy`, `m.cam_pos`. Move the camera body or a mocap body
  for height and pitch.

Design ranges: camera height 0.3–1.6 m, pitch jitter around level to slightly
down, a little roll for handheld photos.

**Image-space augmentations**, applied after rendering (cheap, and also usable
on real photos): motion blur (convolve with a line kernel), Gaussian and shot
noise, JPEG compression at quality 40–90, color jitter, white-balance shift,
vignetting. Apply the same geometric change to the masks, never photometric
ones.

**What the papers found.**

- Tobin et al. 2017 trained an object detector only on rendered images with
  non-realistic random textures. It localized real objects to about 1.5 cm, good
  enough for grasping in clutter, with no real images in training.
- Tremblay et al. 2018 (NVIDIA) used DR for car detection on KITTI. DR-only
  synthetic data gave competitive results, and DR followed by fine-tuning on
  real data beat training on real data alone. This is your E5 setup.
- Prakash et al. 2018, structured domain randomization (SDR): place objects in
  realistic context (cars on roads) while still randomizing appearance. SDR beat
  plain DR and other synthetic datasets on KITTI, and SDR plus real KITTI beat
  real KITTI alone. For you: boxes on the floor near walls, doors at real
  heights, signs beside doors. Randomize looks, keep layout plausible.

### Gotchas

- Randomize the things that differ in real life, not physics. Lighting and
  textures matter more than mass.
- Random textures on the *floor* need care. Too many high-contrast checkers and
  the model learns "floor = busy texture"; real hallway floors are mostly plain
  or speckled. Mix plain colors, photo textures, and some patterns.
- Generating 10k frames at ~3 ms per render plus PNG writing is minutes, not
  hours. The slow part is usually PNG compression; use
  `cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])`.

### Try it (15 min)

Write `make_scene(seed) -> MjModel` with the `MjSpec` code above. Render 9
frames with three seeds and three random light settings each. Tile them into one
image. Same seed twice should give identical pixels; check with
`np.array_equal`.

### Read

- Tobin et al., Domain Randomization (2017): <https://arxiv.org/abs/1703.06907>
- Tremblay et al., Bridging the Reality Gap by DR (2018):
  <https://arxiv.org/abs/1804.06516>
- Prakash et al., Structured Domain Randomization (2018):
  <https://arxiv.org/abs/1810.10093>

______________________________________________________________________

## 3. The sim-to-real gap

**Why it matters for Mini FSD.** The Dec 15 demo runs on real photos, and the E2
and E5 experiments measure exactly this gap. You plan for it in Week 10 (what to
randomize, what to label) and measure it in Weeks 11–12.

### Big picture

A model learns whatever makes sim images easy. If sim floors are always clean
and evenly lit, it learns "uniform region at the bottom = floor" and fails on a
shiny real floor. The gap is the accuracy drop from sim test to real test.

### In the weeds

**Causes, by task.**

- *Floor segmentation:* reflections and specular highlights on polished floors,
  hard shadows, rugs and floor mats, floor-colored walls, baseboards, glass
  doors, clutter at the edges. Sim renders rarely have these.
- *Detection:* real objects (chairs, trash bins, people) have shapes and
  textures your sim boxes do not. Door signs in real buildings vary in font,
  plate color, and mounting height. Real photos have JPEG artifacts and blur.
- *Monocular depth:* scale. A depth network learns scale from cues like object
  sizes and camera height. If sim camera height or field of view differ from the
  phone's, metric depth is off by a factor. Sim depth is perfect; real sensor
  depth (NYUv2) has holes at windows, black surfaces and edges.
- *Camera:* the Pixel 8a applies distortion correction, HDR and sharpening, and
  has a different field of view.

**Usual fixes.**

1. **Domain randomization** (section 2). Cheap, and the baseline of E5.
1. **Match the camera.** Use the calibrated phone intrinsics (section 5) to set
   sim `fovy` and resolution. Many gaps shrink just from this.
1. **Fine-tune on a small real set.** Tremblay et al. found sim + real fine-tune
   beat real alone. Your 50-photo pool from floor 1 is this arm of E5. Keep the
   learning rate low and train briefly, or the model forgets sim variety.
1. **Pseudo-labels.** Run a strong teacher (SAM 2 or SAM 3 for floor masks,
   Depth Anything for depth) on unlabeled real frames, filter by confidence,
   train the small model on them. Depth Anything V2 works this way: synthetic
   labels train a large teacher, which then labels real images for the student.
   Your walking videos are the unlabeled pool; never from test buildings.
1. **Foundation-model features.** A frozen DINOv3 backbone was trained on a huge
   set of real images, so its features describe real-world texture and layout
   well. A light head trained on sim only has to learn "which features mean
   floor." That is the E2 hypothesis. Freezing also stops the backbone from
   overfitting to sim textures.
1. **Scale from geometry.** For depth, recover metric scale from the known
   camera height and the detected floor plane (section 5), instead of trusting
   the network's absolute scale.

**Measure it** with the same metric on sim test, NYUv2 and real test, sliced by
camera height and floor type.

### Gotchas

- Do not tune DR settings on the real test set. Tune on real val (floor 1) only.
- NYUv2 is a different domain too (homes and offices, Kinect at roughly eye
  height, 2011 camera). Good results there do not guarantee good results in
  Northeastern hallways, and the other way around.

### Try it (15 min)

Put 5 Pixel 8a hallway photos next to 5 sim frames at a similar height. List
every visible difference (reflections, shadows, clutter, color cast, blur, field
of view) and rank them by how much each would confuse a floor segmenter. That is
your DR to-do list.

### Read

- Depth Anything V2 (synthetic labels + pseudo-labeled real images):
  <https://arxiv.org/abs/2406.09414>
- DINOv3: <https://arxiv.org/abs/2508.10104>
- Tremblay et al. (sim + real fine-tune result):
  <https://arxiv.org/abs/1804.06516>

______________________________________________________________________

## 4. Datasets and labels

**Why it matters for Mini FSD.** Every model in E1–E6 trains and tests from
these files. Week 10 is when you export the sim set, download NYUv2, and label
the ~150 real photos. Format mistakes here cost days later.

### Big picture

Three kinds of labels: boxes (COCO JSON), floor masks (PNG, one per image),
depth (16-bit PNG, one per image). Plus a small table of room-number text. Split
by building, not by frame, so a test building never leaks into training.

### In the weeds

**COCO JSON for boxes.** One file per split:

```json
{
  "images": [{"id": 1, "file_name": "seed0042_f0007.png", "width": 320, "height": 240}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                   "bbox": [104.0, 52.0, 40.0, 88.0], "area": 3520.0, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "obstacle"}, {"id": 2, "name": "mover"},
                 {"id": 3, "name": "door"}, {"id": 4, "name": "room_sign"}]
}
```

`bbox` is `[x_top_left, y_top_left, width, height]` in pixels, not corners and
not normalized. Category ids start at 1. Ultralytics wants YOLO text files (one
`class cx cy w h` line per box, normalized); it ships a COCO converter. The
`rfdetr` package reads COCO JSON directly; check its README for the expected
folder names.

**Boxes from sim segmentation.** For each obstacle body, take the pixels in the
segmentation image that belong to its geoms and use their min/max as the box.
This gives *visible* (modal) boxes, which is what real labelers draw. Drop boxes
smaller than about 8×8 px or with very few visible pixels.

**Floor masks as PNG.** Single channel, uint8, 0 = not floor, 1 = floor (or 255
if you want to view them). Use 255 as "ignore" for pixels you could not label.
Save with PNG (lossless). Never JPEG.

**16-bit depth PNG.** The common convention (used by many RGB-D datasets):
`uint16` in **millimeters**, `0 = invalid`. Max is 65.535 m.

```python
d_mm = np.where(depth < 20.0, np.round(depth * 1000), 0).astype(np.uint16)
cv2.imwrite("depth.png", d_mm)
back = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
```

Without `IMREAD_UNCHANGED`, OpenCV converts to 8-bit and you lose the data.
Other datasets use other scales (TUM RGB-D uses 5000 per meter, KITTI uses 256),
so write the scale in your dataset README.

**Splitting by seed.** Assign splits by building seed (80/10/10). Frames from
the same building share textures and layout, so a frame-level split leaks: test
accuracy looks great and means nothing. Same rule for real photos: floor 1 =
val, floors 2–3 + other buildings = test. Write a unit test that asserts no seed
or building appears in two splits.

**NYU Depth V2 labeled set.**

- 1,449 aligned RGB + depth pairs from 464 indoor scenes, captured with a
  Kinect. 640×480.
- One file: `nyu_depth_v2_labeled.mat`, about 2.97 GB (2.8 GiB), from
  <https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat>
- `depths` is in-painted (hole-filled) depth in meters; `rawDepths` keeps the
  holes. Use `rawDepths > 0` as the valid mask if you want to score only real
  measurements, and say which one you used.
- `labels` has 894 classes; `floor` is class id 11 (1-based, 0 = unlabeled).
- The standard split is `splits.mat` (795 train, 654 test, 1-based indices) from
  <https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat>

It is a MATLAB v7.3 file, which is HDF5. `scipy.io.loadmat` cannot read it; use
`h5py`. Arrays come out transposed (shapes below were read from the real file):

```python
import h5py, numpy as np

f = h5py.File("nyu_depth_v2_labeled.mat", "r")
f["images"].shape  # (1449, 3, 640, 480) uint8
f["depths"].shape  # (1449, 640, 480) float32, meters
f["labels"].shape  # (1449, 640, 480) uint16
names = ["".join(chr(c) for c in f[ref][:].ravel()) for ref in f["names"][0]]
FLOOR = names.index("floor") + 1  # 11

i = 0
rgb = f["images"][i].transpose(2, 1, 0)  # (480, 640, 3)
depth = f["depths"][i].T  # (480, 640)
floor_mask = f["labels"][i].T == FLOOR

split = scipy.io.loadmat("splits.mat")  # this small file is an old-style .mat
test_idx = split["testNdxs"].ravel() - 1  # to 0-based
```

Convert once to PNGs so the harness reads NYUv2 like your other data. The images
have a thin invalid border from registration and many papers evaluate on a
center crop; pick one crop and state it.

**SAM 2 for floor masks.** SAM 2 (Meta; SAM 2.1 checkpoints from Sep 2024)
segments an object from a few clicks or a box. For a floor: 2–3 positive clicks
on the floor, 1–2 negative clicks on walls or obstacles, then refine. That is
much faster than tracing polygons. `hiera_tiny` (39M params) is the one to try
on the M1 (I did not test it here). Install with
`SAM2_BUILD_CUDA=0 pip install -e .` to skip the CUDA extension. SAM 3 (Nov
2025\) takes text prompts like "floor", which could give a first mask with no
clicks.

**Labeling tool.**

- **Label Studio** (open source, runs locally) has a SAM 2 image backend in
  `label-studio-ml-backend` (`segment_anything_2_image`). Everything stays on
  your laptop. Good fit for 150 photos.
- **CVAT** has a "SAM2 interactor" run through a local agent, but its docs list
  agent functions for CVAT Online and Enterprise only, not self-hosted
  Community. CVAT Online works if you are fine uploading the photos.

Either tool exports COCO for boxes. Export masks as PNG (or COCO RLE and
convert).

**Room-number text.** Draw a `room_sign` box, then store the text as an
attribute on that box (both tools support text attributes), or keep a CSV:
`file_name, box_id, text`. Normalize before scoring (uppercase, strip spaces and
dashes). Report exact-match accuracy on readable signs and count unreadable ones
separately. In sim, the generator already knows the text.

### Gotchas

- Phone photos carry an EXIF rotation tag. Load with
  `PIL.ImageOps.exif_transpose` before labeling and before inference, or masks
  and images will be rotated relative to each other.
- Resize real photos to the model resolution once, with a script, and keep the
  originals. Resize masks with nearest-neighbor, never bilinear.
- Keep label files out of the public repo if the photos include people's faces.

### Try it (20 min)

Open `nyu_depth_v2_labeled.mat` with h5py, export images 0–9 as RGB PNG, floor
mask PNG, and 16-bit depth PNG. Reload one depth PNG and assert the max error
against the original is under 1 mm. Overlay the floor mask on the RGB to check
orientation.

### Read

- COCO data format: <https://cocodataset.org/#format-data>
- NYU Depth V2 page: <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>
- SAM 2 code and checkpoints: <https://github.com/facebookresearch/sam2>
- Label Studio SAM 2 backend:
  <https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image>

______________________________________________________________________

## 5. Camera calibration and the ground-plane homography

**Why it matters for Mini FSD.** Calibrating the Pixel 8a gives you its focal
length and distortion, so sim cameras can match it and real photos can be lifted
into a bird's-eye view. Do the calibration in Week 10 when you shoot the photos;
the bird's-eye projection feeds the costmap in Week 12.

### Big picture

A camera maps 3D points to pixels through its intrinsics `K` (focal lengths
`fx, fy` and principal point `cx, cy`) plus lens distortion. You find them by
photographing a checkerboard whose square size you know. OpenCV finds the
corners, then solves for `K` and distortion.

Once you know `K`, the camera's height `h` and its pitch, every floor pixel
corresponds to exactly one point on the floor. That mapping is a 3×3 homography.
It is how a floor mask becomes a top-down map.

### In the weeds

**Calibration.**

- Print a checkerboard, for example 10×7 squares (9×6 inner corners). Glue it
  flat to cardboard. Measure the square size with a ruler.
- Take 15–25 photos with the same lens (1x), resolution and zoom you will use
  for test photos. Cover the corners of the frame and tilt the board up to about
  45°. OpenCV's tutorial says at least 10.
- Lock focus if the camera app allows it.

```python
pattern = (9, 6)  # inner corners
square = 0.025  # meters
objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square
obj_pts, img_pts = [], []
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
for path in sorted(glob.glob("calib/*.jpg")):
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp)
    img_pts.append(corners)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, gray.shape[::-1], None, None
)
print("RMS reprojection error (px):", rms)
```

**Reprojection error** is the RMS distance between detected corners and where
the solved model projects them. Under about 0.5 px is good; around 1 px is
usable; much more means bad photos (blur, a bent board) or a wrong pattern size.
Drop the worst images and rerun.

If you resize photos from the calibration resolution by factors `sx, sy`, scale
`fx, cx` by `sx` and `fy, cy` by `sy`. Distortion coefficients do not change.
Undistort with `cv2.undistort(img, K, dist)` before any geometry.

**Ground-plane homography.** Robot frame: X forward, Y left, Z up, camera at
height `h`, pitched down by `θ`. Rows of `R` are the optical-frame axes (x
right, y down, z forward) written in the robot frame:

```python
s, c = np.sin(theta), np.cos(theta)
R = np.array([[0, -1, 0], [-s, 0, -c], [c, 0, -s]])
H = K @ np.column_stack(
    [R[:, 0], R[:, 1], -h * R[:, 2]]
)  # floor (X, Y, 1) -> pixel
H_inv = np.linalg.inv(H)  # pixel -> floor
```

Checked against a MuJoCo render (0.5 m high, 20° down, 320×240, fovy 60): floor
point (2.00, 0.40) m projected to pixel (119.5, 98.3), the rendered marker sat
at (118.7, 96.8), and mapping back gave (2.06, 0.42) m. The marker was a sphere,
so its visible center sits slightly above the floor.

For a bird's-eye image, compose `H_inv` with a scale/offset matrix `S` that maps
meters to BEV pixels (say 2 cm per pixel), then
`cv2.warpPerspective(floor_mask, S @ H_inv, (bev_w, bev_h), flags=cv2.INTER_NEAREST)`.
Only pixels below the horizon map to the floor. Points above it (the walls) land
behind the camera or at infinity, so mask them first.

This is also the single-camera scale fix from the design: for a floor pixel, the
homography gives true metric distance. Compare it to the network's depth at the
same pixel and you have a scale factor for the whole image.

### Gotchas

- For phone photos, the pitch is unknown per photo. Hold the phone level with a
  bubble-level app when you shoot test photos, or estimate the horizon from
  vanishing lines. At 0.5 m height, a point 3 m away lands at about 2.3 m or 4.4
  m if your pitch is off by 3° either way, so pitch errors hurt far points the
  most.
- Phones apply their own distortion correction and may crop differently in video
  mode. Calibrate photos and videos separately.
- The homography assumes a flat floor. Ramps and thresholds break it, which is
  fine for a costmap but worth a sentence in the report.

### Try it (20 min)

Calibrate with 15 photos. Report the RMS error. Then put a tape mark on the
floor 2 m ahead of the phone at a known height, take a photo with the phone
level, click the mark, and see how close `H_inv` gets you.

### Read

- OpenCV camera calibration tutorial:
  <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html> (the site
  blocks scripted fetches; it opens in a browser. Source text:
  <https://github.com/opencv/opencv/blob/4.x/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.markdown>)
- OpenCV calib3d reference (`calibrateCamera`, `findHomography`):
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>

______________________________________________________________________

## 6. Classical floor segmentation baseline

**Why it matters for Mini FSD.** E2 compares learned floor segmenters against
this baseline, and the success criterion is that the learned model "clearly
beats" it on real photos. Build it in Week 10 as the first thing the eval
harness scores.

### Big picture

The bottom-center of a forward camera image is almost always floor. Start there
and grow outward, adding neighboring pixels whose color is close to what you
already have. Stop at strong edges (the base of a wall, the outline of a box)
and at the horizon. No training, so it has no sim-to-real gap, but it breaks on
textured floors, shadows and reflections.

### In the weeds

```python
def floor_mask(bgr, lo=3, hi=3, canny=(50, 150), horizon_row=0):
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), *canny)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill wants a 1-px border
    mask[1:-1, 1:-1] = edges > 0  # nonzero = barrier
    mask[: horizon_row + 1, :] = 1  # never fill above the horizon
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(lab, mask, (w // 2, h - 5), 0, (lo,) * 3, (hi,) * 3, flags)
    out = (mask[1:-1, 1:-1] == 255).astype(np.uint8)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
```

Choices to understand:

- **LAB color space.** Lightness (L) is separate from color (a, b), so you can
  loosen L to handle shading while keeping color tight.
- **Floating vs. fixed range.** By default `floodFill` compares each pixel to
  its already-filled neighbor, so it follows slow lighting gradients.
  `FLOODFILL_FIXED_RANGE` compares to the seed color instead, and it stopped
  early on a lit floor in testing. Floating range leaks through weak boundaries,
  which is why the edge barriers and the horizon cap matter.
- **Horizon row.** From the camera pitch and `K`:
  `v_horizon = cy - fy * tan(pitch_down)`. In sim you know it exactly. On real
  photos, use the leveled-phone assumption or skip it.
- **Edges as walls.** Canny edges, dilated so the fill cannot squeeze through
  gaps.

On the simple M1 test scene (checker floor, one box, no walls), floating range
with no horizon cap filled 94% of the image (IoU 0.47). Adding the horizon cap
raised IoU against the MuJoCo floor mask to 0.95. Real hallways will score far
lower, and that is the point of the baseline.

For a stronger baseline: seed from a patch and use its mean and std as the
threshold, grow over SLIC superpixels, or scan each column upward from the
bottom until it hits an edge (a common older robot-nav trick).

### Gotchas

- If the bottom-center pixel is not floor (robot facing a wall, or a box right
  in front), the whole mask is wrong. Check the seed patch and return an empty
  mask with a flag if it looks like an edge.
- Report its latency too. It should be a few ms on the M1, which makes it a
  point on the accuracy-vs-latency plot, not only an accuracy number.

### Try it (15 min)

Run `floor_mask` on 10 real val photos and 10 sim frames. Compute IoU against
the labels. Sort the real photos by IoU and look at the worst three. Write down
why each failed (reflection, shadow, rug, wall color). Those are the cases the
learned model has to beat.

### Read

- OpenCV `floodFill` (Miscellaneous image transformations):
  <https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html>
- OpenCV Canny tutorial:
  <https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>

(docs.opencv.org returns 403 to scripts; the 4.x links redirect to the 4.13.0
pages and open in a browser.)
