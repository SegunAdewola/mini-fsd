# Real photo protocol

How the real photo set (M2) is taken. Use this on the phone while shooting. The
report's method section describes the test set from this page, so if the
protocol changes, change it here first.

## Contents

- [Quick card](#quick-card)
- [How many](#how-many)
- [Getting the height right](#getting-the-height-right)
- [Camera settings](#camera-settings)
- [Pointing the phone](#pointing-the-phone)
- [What to photograph](#what-to-photograph)
- [People](#people)
- [Optional: walking videos](#optional-walking-videos)
- [Afterwards](#afterwards)

## Quick card

- 60 photos per floor: 30 at **20 in**, 30 at **59 in** (lens height).
- Card photo before each block: "F15 · 20 in".
- Main 1x lens, landscape, 4:3, no zoom, no Portrait, no Night Sight.
- Floor visible in almost every photo. Horizon level.
- Move about 6 ft or turn to a new view between shots.
- About 8 photos per floor with people in them, faces turned away or far off.
- No identifiable faces.

## How many

About 50 usable photos per floor, half at each height. Take about 60, since some
come out blurred or nearly the same as another.

| Floor | Use                                                           | Target |
| ----- | ------------------------------------------------------------- | ------ |
| 15    | Validation: pick models here; pool for the 50-photo fine-tune | 50     |
| 14    | Test: same layout as 15, unseen floor                         | 50     |
| 2     | Test: a layout the models have never seen                     | 50     |

One building only (decided 2026-11-22). Floors 14 and 15 share a layout, so
floor 2 is the test of an unseen layout; NYUv2 test covers unseen buildings.

Once the test photos are copied over, nobody looks at model results on them
until the final run.

## Getting the height right

Heights are lens heights above the floor. Within 4 in either way is fine: the
sim trains on every height from 12 to 63 in (0.3 to 1.6 m), so the photos only
need to land near the two targets.

| Target        | Inches                | Metres |
| ------------- | --------------------- | ------ |
| Robot height  | 20 in (19.7)          | 0.5 m  |
| Handheld      | 59 in                 | 1.5 m  |
| Allowed error | about 4 in either way | 0.1 m  |

**20 in, with a string.** Cut a 20 in string and tie a key or other small weight
to one end. Hold the other end at the phone's lens and lower the phone until the
weight just touches the floor. Or measure once where 20 in falls on your leg
(often just below the knee) and crouch so the lens is level with it.

**59 in, with a body mark.** At home, stand with your back to a wall, mark 59 in
with the tape measure, and note where it lands on you: chin, mouth or eyes. In
the building, always hold the lens at that spot. The usual arms-out photo
position is lower than it feels.

**Keep the heights apart.** Shoot all of one height on a floor, then all of the
other. Before each block, photograph a handwritten card such as "F15 · 20 in" or
"F15 · 59 in". The cards let the photos be sorted by floor and height
automatically later; they are then deleted.

## Camera settings

The same for every photo, and for the checkerboard calibration photos (V4).
Calibration only holds for one lens at one zoom.

- Main 1x lens only. No 0.5x ultrawide, no zoom.
- Landscape, 4:3, full resolution (the Pixel's default 12 MP).
- Off: Portrait, Night Sight (it can switch on by itself in dim areas), Top Shot
  and motion photos.
- On: grid and level (Settings, Framing hints).
- No cropping, filters or edits afterwards.

## Pointing the phone

- At 20 in: about level, like the robot's camera. A slight tilt down is fine.
- At 59 in: tilt down 0 to 25 degrees, so the floor fills roughly the bottom
  third of the frame. This matches the tilt range the sim trains on (D1).
- Keep the horizon level with the level indicator; tilt sideways no more than
  about 5 degrees.
- Floor should show in almost every photo. Floor segmentation is the main thing
  the models are judged on.

## What to photograph

Per floor, roughly:

| What                                                         | About | Why                                             |
| ------------------------------------------------------------ | ----- | ----------------------------------------------- |
| Corridors and the elevator lobby, looking along them         | 10    | Closest to the sim                              |
| Open areas, desk clusters, around the core                   | 10    | Open plan, which the generator doesn't make yet |
| Door signs with readable room numbers, from 3 to 10 ft       | 8     | Sign reading                                    |
| Glass walls and doors, on purpose                            | 5     | Known failure case for depth                    |
| Things on the floor: chairs, bins, bags, boxes               | 5     | Obstacle dodging                                |
| Hard cases: shiny floor, carpet, bright windows, dim corners | 4     | Honest coverage                                 |
| People walking or standing, from behind, side-on or far off  | 8     | Person detection (moving obstacles)             |

Move about 6 ft (2 m) between shots, or turn to face a new direction. Spread the
photos over the whole floor, not just near the elevators.

## People

People are part of the test now: the robot has to detect them as obstacles. Take
about 8 photos per floor with people in them, on top of the shot list above.
Keep them unidentifiable:

- Photograph people from behind, side-on, or far enough away (about 15 ft or
  more) that a face is a few pixels.
- If someone is close and facing you, wait for them to pass or skip the shot.
- Ask before photographing anyone up close, even from behind. Don't photograph
  anyone who objects, and delete the photo if they ask.
- Before copying the photos over, blur any face that is still recognisable, or
  delete the photo. Check at full size, not on the phone's thumbnail.

The photos are never committed to GitHub, but some may appear in the report. For
the rest of the shot list, frame people out as before.

## Optional: walking videos

Two or three slow walking videos at 20 in, about 30 s each. They give unlabeled
frames for depth distillation and a clip for the demo. Same lens and settings as
the photos.

## Afterwards

Copy the originals over USB, or use Google Photos, Download original. Shared or
compressed copies lose the EXIF data, which holds the focal length. Put them in:

```text
data/real/raw/floor-02/
data/real/raw/floor-14/
data/real/raw/floor-15/
```

`data/` is gitignored, so the photos are never committed. Before any labeling,
check the count per floor and height, the resolution, and that EXIF is intact.
