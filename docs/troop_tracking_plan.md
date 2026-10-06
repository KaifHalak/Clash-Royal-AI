# Troop Tracking & Game State Plan

## Goal

Give an ML model the complete state of the game as text, so it can decide what to play. The state includes:

- **User data:** cards in hand, next card, current elixir, tower HP.
- **Enemy data:** estimated elixir, troops on the field.
- **Battlefield:** every troop (enemy and user) with its type, level, position and status.

This document covers the part that doesn't exist yet: tracking troops over time and turning them into that text state. Troop detection (`src/troop_extractor.py`) already exists and is the starting point.

---

## 1. MVP scope

| In the MVP | Left out of the MVP (on purpose) |
|---|---|
| Troop type, identified once per track | Exact HP; only alive or dead |
| Troop level, read once per track | Spells |
| Position as a grid tile plus zones | Individual swarm units; one entry per deploy |
| Alive or dead, and time since deploy | ID swap correction and re-identification |
| Deploy events and an enemy elixir estimate | Dash, jump, invisibility and knockback handling |
| User troops, seeded from our own deploys | Enemy card cycle tracking |

---

## 2. Per-frame pipeline

For every captured frame:

1. Crop the frame to the arena.
2. Detect enemy badges (red) and user badges (blue).
3. Update the enemy tracker and the user tracker with those detections.
4. Identify and read the level of any newly confirmed enemy track.
5. Emit deploy events for new enemy tracks and update the enemy elixir estimate.
6. Convert every track's position from pixels to grid tiles and zones.
7. Build the text state and pass it to the model.

---

## 3. Troop detection (existing)

`TroopExtractor` finds enemy badges by colour thresholding (red fill plus white border), joins the pieces, and filters blobs by size, squareness, fill and a digit check.

**Known issues to fix before tracking depends on it:**

- **Noise merging into badges.** White noise pixels near a badge get joined to it by the CLOSE step, so the badge fails the shape checks and is missed. Proposed fix: anchor candidates on the red fill (much more specific than white) and use the white border only to confirm the box. Removing small components before joining is a backup option.
- **Princess tower bar detected as a badge.** Needs the tower regions added to `TOWER_IGNORE_REGIONS`. The king tower coordinates from `static_elements_extractor.py` fall outside the current arena crop, so the princess tower coordinates are needed instead, converted from full-frame ratios to arena ratios.
- **Adjacent badges merging.** Two troops close together can become one blob. The tracker tolerates this (section 5), but fewer merges means fewer lost tracks. The red-fill anchor above should help here too.
- **User badges.** Reuse the same pipeline with blue colour thresholds.

---

## 4. Enemy tracking: the normal case

Tracking is done by detection: detect every frame, then match detections to existing tracks.

**Each track stores:**

- ID
- Label (troop type) and level
- Last position (badge centre, in pixels)
- Last time seen, and number of consecutive misses
- Status: tentative, confirmed, uncertain, dead

**Matching each frame:**

1. **Distance table.** Measure the distance from every track's last position to every new detection.
2. **Global assignment.** Pick the pairing with the smallest total distance across all tracks at once (`scipy.optimize.linear_sum_assignment`), instead of matching one track at a time. Greedy matching lets two tracks claim the same detection.
3. **Maximum distance.** Reject any pairing further than the allowed distance, so a track can't jump across the arena to grab an unrelated troop.
4. **Update.** Matched tracks take the new position. Labels are never recomputed; they stay with the track.

**Why position is enough most of the time:** between two frames a troop moves a few pixels, while neighbouring troops are usually tens of pixels apart. Each badge's new position is much closer to its own old position than to anything else. This holds as long as each troop moves less than about half the gap between two troops per frame.

**Track in pixels, not tiles.** Two troops can share a tile, and a troop walking along a tile boundary flips between cells. Tiles are only for output (section 10).

**No hard-coded movement paths.** Troops don't always walk to the bridge. For example, enemy troops turn toward a Hog Rider that jumped the river. If prediction is added later, it should be constant velocity from observed movement only, with no game-path assumptions. Start without prediction and add it only if ID switches show up.

**Track lifecycle:**

- **New detection with no match:** create a *tentative* track. Confirm it only after it's been matched in several consecutive frames. This filters out one-frame noise before it becomes a ghost troop or a false deploy event.
- **Track with no match:** keep it alive and count the miss. Mark it *dead* only after it's been missing longer than the miss tolerance.
- **All limits in time, not frames.** Capture fps varies, so the allowed distance and miss tolerance should be in seconds or pixels per second.

---

## 5. Enemy tracking: failure cases

### 5.1 Detection missed for several frames

The troop moves further than the normal allowed distance while it's not detected.

- Let the allowed distance grow with time missing: `allowed distance = max troop speed × time since last seen`.
- After a long gap, position is only a rough guide. Match lost tracks to new detections by identity, within the allowed distance.
- A detection that fits no lost track becomes a new tentative track. A track missing for too long becomes dead.

### 5.2 Troops overlapping or crossing

The nearest pairing can be the wrong one, so the two tracks swap labels.

- **Ambiguity test.** Compare the best pairing with the second best. If `second best ÷ best` is below a threshold (start around 2), the match is ambiguous. Use a ratio, not a pixel difference, so it works at any scale.
- **Early warning.** When two tracks get close, mark the pair as *at risk*.

### 5.3 Signs that tracking has gone wrong

Any of these marks the track as *uncertain*:

1. The best match isn't clearly better than the second best (ambiguity test).
2. The match is further than the allowed distance.
3. The detection count doesn't match the track count (a merge, a miss, or something new).
4. Two tracks are very close together.
5. A troop's HP went up (only once HP tracking exists; HP only goes down).

### 5.4 Recovery for uncertain tracks (post-MVP)

- Keep the label, but flag it as unconfirmed. Don't delete or rename the track.
- Re-identify once the track's troop crop no longer overlaps another track's crop. Overlapping crops contain parts of both troops, so identification is unreliable until then.
- If identification agrees, clear the flag. If it disagrees, swap or correct the label.
- If the uncertain tracks are the same troop type, skip all of this. A swap between two identical troops doesn't change the game state.

---

## 6. Identification

- Identify each enemy track **once**, when it's confirmed. At that point the troop is usually alone, so the crop is clean.
- Take a majority vote over the first few frames so one bad read doesn't mislabel the track.
- Run identification in a background thread so the tracker isn't blocked by the model.
- Only re-identify after an uncertain event (section 5.4), never every frame. Identifying every frame is too slow, flickers on bad reads, and can't separate two troops of the same type anyway.

**Distinguishing troops:** position keeps troops apart frame to frame. Identity (e.g. Knight vs Valkyrie) is used to correct swaps after merges or crossings. The badge doesn't help much: the level is usually the same for all of one player's troops, and the HP bar only appears once a troop is damaged.

---

## 7. Level

- The digit inside the badge is the troop level. Read it once per track with OCR (`read_badge_number`); it doesn't change.
- Take a majority vote over a few frames, like identification.

---

## 8. Deploy events and enemy elixir

**Deploy events** must be counted exactly once per card played, because the elixir estimate depends on them.

- A new track counts as a deploy only once it's confirmed (tentative tracks filter out noise).
- A lost-then-recovered track must not count as a second deploy (the miss tolerance handles this).
- Multi-unit cards (Skeleton Army, Barbarians, Minion Horde) create several tracks at once. Group tracks that appear at nearly the same time and place into one deploy.
- Units that appear without a deploy (spawners, death spawns) must not count as deploys. See section 12.

**Enemy elixir estimate:**

`elixir = starting elixir + regeneration over time − cost of each deploy`, capped at 0 and 10.

- The regeneration rate changes during double elixir and overtime. Use the game clock from the timer extractor to switch rates.
- Confirm the exact starting elixir, regeneration rate and timings before relying on them.
- Needs a card table with each card's elixir cost.
- **Spells are excluded from the MVP, so the estimate is an upper bound.** The enemy still pays for spells we don't see. Either label it as an upper bound, or later detect "a spell was cast" and subtract an average cost.

---

## 9. User troop tracking

The user's troops are deployed by us, so we know the card, the time and the position. The same tracker code is used, with two differences.

- **Seeded tracks.** Each deploy creates a track directly, with the identity already known. No identification and no tentative phase.
- **Blue badges.** Following the troop after deploy still needs detections every frame, from the blue-badge version of the detector.

**Details:**

- **Wait for the troop to appear.** Troops take about a second to deploy. Create the track as *pending* at the deploy point, attach the first blue badge that appears nearby, then track normally. If nothing appears within a few seconds, drop the pending track.
- **Deploy point vs badge position.** The deploy point is the ground position; the badge appears one to two tiles higher. Apply the ground-to-badge offset (section 10) to predict where the first badge will appear.
- **Multi-unit cards.** Attach all badges near the deploy point to one deploy group.
- **Separate trackers per team.** Badge colour already separates the teams, so enemy and user tracks never compete in matching. Write the tracker once; the two instances only differ in badge colours and in how tracks start (unmatched detections for the enemy, deploy events for the user).

---

## 10. Pixels to grid

The arena is commonly cited as **18 tiles wide × 32 tiles tall**, with the river across the middle. Verify this against screenshots.

**Calibration (once):**

1. **Get a grid screenshot.** When a card is dragged before placing it, the game draws the tile grid on the arena. Screenshot that to get exact tile edges.
2. **Measure the corners.** Find the top-left corner of tile (0, 0) and the bottom-right corner of tile (17, 31) in pixels. Store them as fractions of the frame, like `static_elements_extractor.py` does, so it works at any resolution.
3. **Check the tile spacing.** The camera has a slight tilt. If tiles near the top look smaller than tiles near the bottom, use a perspective transform from the four arena corners (`cv2.getPerspectiveTransform`). If spacing is even, a linear mapping is enough:
   - `tile_x = (px − arena_left) ÷ tile_width`
   - `tile_y = (py − arena_top) ÷ tile_height`

   The perspective transform handles both cases, so it's the safe default.

**Use one coordinate system.** Badges are found in the cropped arena image. Either add the crop offsets back before converting, or calibrate the corners in cropped-arena coordinates. Mixing the two is the same bug as the tower ignore regions.

**Convert the ground position, not the badge.** The badge floats above the troop. Converting the badge centre directly puts every troop one or two tiles too far up. Add a vertical offset in badge-size units (like the troop crop) to estimate where the troop stands. Air units float higher, so they need a different offset once identified. For the MVP, one offset for all troops is fine.

**Output to the model:**

- **Tile:** e.g. (4, 12)
- **Lane:** left or right, from `tile_x` relative to the centre
- **Side:** enemy or user, from `tile_y` relative to the river
- **Landmarks:** e.g. "at left bridge", "attacking left princess tower". Bridges and towers are at fixed tiles, so set their positions once from the grid screenshot.

---

## 11. Game state output

An example of the level of detail. This is not a final format.

```
time_remaining: 2:14 (normal elixir)
my_elixir: 6
my_hand: [Hog Rider, Musketeer, Ice Spirit, Cannon]  next: Fireball
my_towers: left 2130/2534, right 2534/2534, king inactive
enemy_elixir_estimate: ~4 (upper bound; spells not tracked)
enemy_troops:
  - Knight, lvl 11, tile (5, 14), left lane, enemy side, at bridge, alive, deployed 3s ago
  - Musketeer, lvl 11, tile (8, 3), left lane, enemy side, behind king tower, alive, deployed 1s ago
my_troops:
  - Hog Rider, lvl 11, tile (14, 16), right lane, crossing bridge, alive
```

---

## 12. Build order

1. Fix the detection issues in section 3 (red-fill anchor, princess tower ignore regions).
2. Enemy position tracker: global assignment, max distance, miss tolerance, tentative tracks, time-based limits.
3. Debug view: draw track IDs on replay frames and check that IDs stay stable.
4. Identification and level reading once per confirmed track.
5. Deploy events with swarm grouping.
6. Enemy elixir estimate from deploys and the timer.
7. Pixel-to-grid calibration and zones.
8. User troop tracking (blue badges, seeded tracks).
9. Assemble the text state.
10. Validate on recorded replays.

---

## 13. Known edge cases (not handled in the MVP)

**Troops that appear without a deploy.** These break the "new track = deploy" rule and would inflate the enemy elixir estimate.

- Spawners: Tombstone, Goblin Hut, Furnace, Witch, Night Witch.
- Death spawns: Golem into Golemites, Lava Hound into pups, Battle Ram into Barbarians, Elixir Golem splitting.
- Graveyard: a spell (excluded) that spawns troops on the user's side.
- Miner: surfaces anywhere, including the user's side.
- Clone: duplicates appear on top of the originals, same type and position. The worst case for position matching.

A later rule: a new track near an existing spawner or splitting unit is expected, not a deploy.

**Sudden jumps and disappearances.**

- Dash and jump units: Bandit, Mega Knight, Golden Knight, Prince's charge, Hog Rider's river jump. Can exceed the allowed distance in one frame.
- Invisibility: Royal Ghost, Archer Queen's ability, Tesla hiding underground. Looks like a death followed by a new troop.
- Knockback and pulls: Log, Fireball, Snowball, and especially Tornado, which pulls troops into one big merge.
- Speed changes: Rage breaks the max-speed limit; Freeze drops velocity to zero.

These fit naturally as per-card properties ("can dash", "can go invisible") on identified tracks.

**Things that break detection.**

- Buildings (Cannon, Inferno Tower) have badges but never move. A stationary track shouldn't be treated as stuck.
- Spell colours (Rage, Poison) can shift badge colours out of the thresholds.
- Emotes can cover badges near the king towers.
- Air units flying over ground units cause more badge overlaps.

**Death vs missed detection.** Both look like an unmatched track. A dead troop lingers for the miss tolerance before being marked dead. Fine for board state, but inaccurate if deaths are counted.

**Two identical troops swapping.** Two same-type, same-level, undamaged troops that cross or merge can't be told apart. This is acceptable: they're interchangeable for game state.

---

## 14. Open questions and things not discussed yet

- **What kind of model?** An LLM reading text, or a model trained on our own data. This affects the state format, how much detail to include, and how often to send updates.
- **Update frequency and latency.** How often the state is sent to the model, and how long a decision can take before it's too late to act on.
- **Capture fps.** Tracking by position assumes troops move only a few pixels between frames. At around 1 fps or lower, troops cross each other between frames and matching gets much harder. Measure the real capture rate.
- **How user deploys are captured.** Seeding user tracks assumes we know the card, time and position of each deploy. That's true if the bot plays; if a human plays, the deploy has to be detected (e.g. from the hand changing and elixir dropping).
- **Enemy card cycle.** After seeing all 8 enemy cards, their hand and next card can be estimated. Not in the MVP, but deploy events make it possible later.
- **Spell detection.** Needed for an accurate enemy elixir estimate and for Graveyard.
- **Card data table.** Elixir cost, speed, air or ground, and special behaviours for each card. Needed for the elixir estimate, allowed distances and edge-case handling.
- **Match start and end.** Detecting when a match starts (to reset trackers and elixir) and ends.
- **Performance budget.** Detection, tracking, identification and OCR all have to fit within the frame interval. Model loading must happen once.
- **Different devices and resolutions.** Crop values and calibration are currently tuned for one frame size. Everything should be stored as ratios.
- **Testing.** Record matches with the in-game replay feature, hand-label troops for a short stretch, and measure ID switches, missed troops and false deploys. Without this it's impossible to tell whether a change helped.
- **Champions and evolutions.** Abilities and evolved cards may look and behave differently from the base card.

---

## 15. Alternative approaches (for later)

- **Trained object detector (e.g. YOLO) plus ByteTrack.** One model detects and identifies troops from the sprite, with or without a badge. ByteTrack is designed to avoid losing tracks on low-confidence detections. Needs labelled data, which can be bootstrapped by auto-labelling frames with the current badge pipeline and correcting them. The strongest long-term option if badge detection stays the bottleneck.
- **Background subtraction.** The arena background is static, so subtracting it finds troops even without a badge. Useful as a backup detector, but spell effects add a lot of noise.
- **Game-model tracker.** Predict each troop's movement from its card stats and game rules, corrected by observations (a particle filter). Powerful, but far more work than the MVP needs.
