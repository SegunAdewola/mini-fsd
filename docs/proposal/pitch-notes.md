# Mini FSD: pitch notes (5 minutes, Nov 10)

Talk, don't read. Times are rough targets. Optional image to show: one Robin
floor plan of a campus floor.

## 1. The problem (0:00–0:45)

"My project is Mini FSD: Tesla's Full Self-Driving idea, scaled down to a robot
inside a building. You tell it 'go to room XYZ,' and it plans a route and drives
there, using only cameras to see obstacles. The robot is in simulation, but the
vision has to work on real photos, and everything has to run on a small
computer."

- Why it matters: delivery, hospital and warehouse robots do this every day on
  small onboard computers.
- Speed and safety depend on how fast the vision model is. At 2 m/s, a model
  that takes 0.1 s per image means 20 cm of driving blind.
- So I treat my 8 GB laptop, under strict limits, as the robot's computer, and
  aim for 20 images per second.

## 2. Data and approach (0:45–2:45)

**Data, three sources:**

- About 10,000 simulated images from random office floors in MuJoCo. The
  simulator labels every pixel for free: floor, obstacles, people, doors, room
  signs, depth.
- About 160 real photos I take on my Pixel: the 3 floors of the Vancouver campus
  plus 2–3 other buildings. Floor 1 is for choosing models; floors 2–3 and the
  other buildings are the test set.
- NYU Depth V2: real indoor photos with measured depth.
- I also rebuild the campus floors in the simulator from the building's floor
  plans, so "go to room XYZ" uses real room numbers.

**How it works:**

- ROS 2 connects the simulator, the vision models, and Nav2, which plans the
  route and steers.
- The vision output becomes a top-down obstacle map, a small version of Tesla's
  bird's-eye view.

**Models: I compare instead of guessing.**

- Detection: YOLO26n vs. RF-DETR.
- Floor: EfficientViT vs. a frozen DINOv3 backbone.
- Depth: Depth Anything 3 vs. V2 vs. a smaller copy vs. two-camera stereo. The
  robot uses whichever wins; the photo demo uses the best single-camera model.
- Then: separate models vs. one shared network, like Tesla's HydraNet.
- Each one is timed under three hardware limits, roughly a Jetson Orin NX, an
  Orin Nano, and a Raspberry Pi 5.

**Baselines:** a classical floor finder with no learning, and the same network
trained without random textures and lighting.

**Extra, only if time permits:** 3–5 robots sharing the building, with a central
manager that assigns trips.

## 3. What success looks like on Dec 15 (2:45–3:45)

- On your unseen indoor photos (hallways, classrooms, offices, lobbies, from
  robot height to handheld), it highlights the safe floor, draws boxes, shows
  depth, and reads clear room numbers.
- Targets:
  - floor IoU of at least 0.80 on my real test photos, beating the classical
    baseline;
  - at least 20 images per second;
  - at least 90% of trips reaching the room in buildings the robot has never
    seen.
- Out of scope: outdoors, stairs, very dark images.

## 4. Biggest risk (3:45–4:30)

- **Models trained on simulated images often fail on real photos.** I test on
  real photos early (week 10). The fallbacks are stronger randomization,
  automatic labels on extra real frames, and fine-tuning on 50 real photos.
- **Second risk: ROS 2 on an 8 GB Mac.** I test it this week, with Docker as the
  backup.
- **If I fall behind,** I cut in a set order, starting with the fleet.

## Likely questions

- **Why simulation?** I agreed it with you. It lets me spend the time on vision,
  and the simulator gives exact labels for free.
- **Why not just use the biggest model?** It's too slow for the robot's
  computer, and published speed numbers depend on the hardware, so I measure on
  mine.
- **How do you know it works on real photos?** It's tested on real photos from
  floors and buildings it was never tuned on, plus NYU Depth V2.
- **Why not stereo for everything?** The demo photos are single images, so the
  photo demo has to use single-camera depth. In simulation I compare both, and
  the robot uses the winner.
- **What's the CV work vs. the robotics?** About half the time goes to the data,
  the model comparisons, real-photo testing, and speeding models up. Nav2 does
  the route planning out of the box.
- **Isn't one office building a narrow test?** Yes, which is why the test set
  also includes photos from 2–3 other buildings.
