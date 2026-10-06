import itertools
import math

# Test script for the enemy position tracker (see docs/troop_tracking_plan.md, section 4).
# Badge coordinates are hard-coded below instead of coming from TroopExtractor,
# so we can check the tracking logic on its own. Once it works, convert to a class.

# ================= CONFIG =================
MAX_SPEED_PX_PER_S = 40       # fastest a badge can move; allowed distance = this * time since last seen
MIN_MATCH_DISTANCE_PX = 10    # floor for the allowed distance, so a troop that barely moved still matches
CONFIRM_HITS = 3              # frames a new track must be matched in before it's confirmed (filters noise)
MISS_TOLERANCE_S = 1.0        # how long a confirmed track survives without a match before it's dead


# ================= TEST DATA =================
# Each frame: (timestamp in seconds, [badges]), 5 fps.
# Badge = (x, y, side, truth_label). x, y is the badge's top-left corner, same as find_badges.
# truth_label is ONLY used to check the result at the end; the tracker never looks at it.
#
# What happens in this sequence:
# - knight and valkyrie walk down side by side, 30 px apart
# - noise: a one-frame false detection at t=0.6 (should never be confirmed)
# - knight is missed at t=1.2 and t=1.4, then comes back (should keep the same ID)
# - deployed: a new troop appears at t=1.6 (should get a new ID, confirmed after 3 frames)
# - valkyrie disappears after t=2.2 (should be marked dead once MISS_TOLERANCE_S has passed)
FRAMES = [
    (0.0, [(100, 50, 12, "knight"), (130, 50, 12, "valkyrie")]),
    (0.2, [(100, 54, 12, "knight"), (130, 54, 12, "valkyrie")]),
    (0.4, [(100, 58, 12, "knight"), (130, 58, 12, "valkyrie")]),
    (0.6, [(100, 62, 12, "knight"), (130, 62, 12, "valkyrie"), (300, 200, 12, "noise")]),
    (0.8, [(100, 66, 12, "knight"), (130, 66, 12, "valkyrie")]),
    (1.0, [(100, 70, 12, "knight"), (130, 70, 12, "valkyrie")]),
    (1.2, [(130, 74, 12, "valkyrie")]),
    (1.4, [(130, 78, 12, "valkyrie")]),
    (1.6, [(100, 82, 12, "knight"), (130, 82, 12, "valkyrie"), (250, 60, 12, "deployed")]),
    (1.8, [(100, 86, 12, "knight"), (130, 86, 12, "valkyrie"), (250, 63, 12, "deployed")]),
    (2.0, [(100, 90, 12, "knight"), (130, 90, 12, "valkyrie"), (250, 66, 12, "deployed")]),
    (2.2, [(100, 94, 12, "knight"), (130, 94, 12, "valkyrie"), (250, 69, 12, "deployed")]),
    (2.4, [(100, 98, 12, "knight"), (250, 72, 12, "deployed")]),
    (2.6, [(100, 102, 12, "knight"), (250, 75, 12, "deployed")]),
    (2.8, [(100, 106, 12, "knight"), (250, 78, 12, "deployed")]),
    (3.0, [(100, 110, 12, "knight"), (250, 81, 12, "deployed")]),
    (3.2, [(100, 114, 12, "knight"), (250, 84, 12, "deployed")]),
    (3.4, [(100, 118, 12, "knight"), (250, 87, 12, "deployed")]),
]


# ================= TRACKER =================
next_track_id = itertools.count(1)


def badge_centre(badge):
    x, y, side = badge[:3]
    return (x + side / 2, y + side / 2)


def allowed_distance(track, now):
    """How far the track's badge could have moved since it was last seen."""
    return max(MIN_MATCH_DISTANCE_PX, MAX_SPEED_PX_PER_S * (now - track["last_seen"]))


def match_tracks(tracks, centres, now):
    """Greedy matching: take the closest allowed (track, detection) pair, repeat with what's left.
    Returns a list of (track_index, detection_index)."""
    
    
    # Track: 1 troop. 
    # Detection: 1 badge
    
    # So we are comparing each troop (track) with all the badges identified (detections) to see if which of the new detections is the troop
    
    # Every pairing that's within the track's allowed distance
    candidate_pairs = []
    for track_index, track in enumerate(tracks):
        limit = allowed_distance(track, now)
        for detection_index, centre in enumerate(centres):
            
            # Find the distance between the last known position of the track and the centre of the new detection 
            distance = math.dist(track["position"], centre)
            if distance <= limit:
                candidate_pairs.append((distance, track_index, detection_index))

    # Closest first (ascnending order / smallest first); skip any pair whose track or detection is already taken
    candidate_pairs.sort()
    used_tracks, used_detections = set(), set()
    matches = []
    for distance, track_index, detection_index in candidate_pairs:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))
    return matches


def update_tracks(tracks, badges, now):
    """One tracker step. Returns (tracks, assignments, newly_confirmed_ids, newly_dead_ids).

    assignments: {detection_index: track_id}, which track each detection went to
    """
    centres = [badge_centre(badge) for badge in badges]
    matches = match_tracks(tracks, centres, now)
    matched_track_indexes = {track_index for track_index, _ in matches}
    matched_detection_indexes = {detection_index for _, detection_index in matches}

    assignments = {}
    newly_confirmed_ids = []
    newly_dead_ids = []

    # 1. Matched tracks: move to the new position
    for track_index, detection_index in matches:
        track = tracks[track_index]
        track["position"] = centres[detection_index]
        track["side"] = badges[detection_index][2]
        track["last_seen"] = now
        track["hits"] += 1
        if track["status"] == "tentative" and track["hits"] >= CONFIRM_HITS:
            track["status"] = "confirmed"
            newly_confirmed_ids.append(track["id"])
        assignments[detection_index] = track["id"]

    # 2. Unmatched tracks: tentative ones were probably noise, confirmed ones get a grace period
    surviving_tracks = []
    for track_index, track in enumerate(tracks):
        if track_index in matched_track_indexes:
            surviving_tracks.append(track)
        elif track["status"] == "tentative":
            continue                                                   # drop silently
        elif now - track["last_seen"] > MISS_TOLERANCE_S:
            track["status"] = "dead"
            newly_dead_ids.append(track["id"])
        else:
            surviving_tracks.append(track)                             # missed, but still alive

    # 3. Unmatched detections: start new tentative tracks
    for detection_index, badge in enumerate(badges):
        if detection_index in matched_detection_indexes:
            continue
        new_track = {
            "id": next(next_track_id),
            "position": centres[detection_index],
            "side": badge[2],
            "first_seen": now,
            "last_seen": now,
            "hits": 1,
            "status": "tentative",
        }
        surviving_tracks.append(new_track)
        assignments[detection_index] = new_track["id"]

    return surviving_tracks, assignments, newly_confirmed_ids, newly_dead_ids


# ================= RUN =================
if __name__ == "__main__":
    tracks = []
    label_to_track_ids = {}       # truth label -> every track ID its detections were given
    confirmed_ids = set()
    dead_ids = set()

    for now, badges in FRAMES:
        tracks, assignments, newly_confirmed_ids, newly_dead_ids = update_tracks(tracks, badges, now)
        confirmed_ids.update(newly_confirmed_ids)
        dead_ids.update(newly_dead_ids)

        for detection_index, track_id in assignments.items():
            label = badges[detection_index][3]
            label_to_track_ids.setdefault(label, set()).add(track_id)

        # Print this frame: every live track, plus events
        track_texts = []
        for track in tracks:
            missed = " (missed)" if track["last_seen"] < now else ""
            x, y = track["position"]
            track_texts.append(f"#{track['id']} {track['status']} ({x:.0f},{y:.0f}){missed}")
        events = ""
        if newly_confirmed_ids:
            events += f"  CONFIRMED {newly_confirmed_ids}"
        if newly_dead_ids:
            events += f"  DEAD {newly_dead_ids}"
        print(f"t={now:.1f}  " + " | ".join(track_texts) + events)

    # ================= CHECKS =================
    print()
    checks = []
    for label in ["knight", "valkyrie", "deployed"]:
        ids = label_to_track_ids.get(label, set())
        checks.append((f"{label} kept one ID the whole time {sorted(ids)}", len(ids) == 1))
        checks.append((f"{label} was confirmed", bool(ids & confirmed_ids)))

    noise_ids = label_to_track_ids.get("noise", set())
    checks.append(("noise was never confirmed", not (noise_ids & confirmed_ids)))

    valkyrie_ids = label_to_track_ids.get("valkyrie", set())
    checks.append((f"only valkyrie died {sorted(dead_ids)}", dead_ids == valkyrie_ids))

    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {description}")
