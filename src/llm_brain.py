import ast
import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from groq import Groq

from board_state import GAME_COLUMNS, GAME_ROWS, legend
from rule_based_brain import CARDS, OUR_FIRST_ROW, PLAY_COOLDOWN, UNIT_TYPES

try:
    from dotenv import load_dotenv     # optional: lets GROQ_API_KEY live in a .env file
    load_dotenv()
except ImportError:
    pass

# LLM decision maker: game state -> Groq chat model -> JSON -> action (or wait).
# Same action format as RuleBasedBrain, so main.py can swap between them.
#   - the request runs in a background thread, so the frame loop (and YOLO tracking) never stalls
#   - the reply is asked for as JSON (Groq JSON mode), and is still parsed defensively:
#     a broken / missing / nonsense reply just means "wait", it never crashes the bot
#   - the chosen card is re-checked against the CURRENT hand + elixir before it is played,
#     because the state has moved on by the time the reply arrives

# ================= MODEL =================
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")   # any Groq model that supports JSON mode
TEMPERATURE = 0.3
MAX_TOKENS = 500          # the reply is one small JSON object, but reasoning models "think" first
REQUEST_TIMEOUT = 10      # seconds; a reply older than this is useless anyway

# ================= TUNING =================
REQUEST_INTERVAL = 5.0    # seconds between requests (Groq's free tier is rate limited, ~30 requests/min)
RATE_LIMIT_BACKOFF = 10   # seconds to pause after a rate-limit / API error
RECENT_PLAYS = 5          # our last N plays are shown to the model so it doesn't repeat itself
MIN_CARD_COST = min(card["cost"] for card in CARDS.values())


def model_options(model):
    """Extra request options per model. gpt-oss reasons before answering: at its default effort it runs
    out of MAX_TOKENS mid-thought and Groq rejects the reply, so keep it low (also faster, ~0.7 s)."""
    return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}

# ================= SYSTEM PROMPT =================
# Per-card advice for our deck; cost + role come from rule_based_brain.CARDS
CARD_TIPS = {
    "giant": "slow, very tanky, ONLY attacks buildings/towers. Your win condition. May use it to defend. Best played at the back (behind your king tower) when elixir is high, then support it. Can also use to defend to stall out enemy troops",
    "mini_pekka": "fast, huge single-target melee damage, ground only (cannot hit air). "
                  "Best answer to giant, knight and mini-pekka. Weak to swarms and minions.",
    "knight": "cheap tanky melee, ground only. Solid defender vs ranged and melee troops, "
              "and a cheap mini-tank in front of a push.",
    "musketeer": "long-range single target, hits air AND ground. Place it behind a tank or behind your "
                 "princess tower. Dies to fireball.",
    "archers": "2 ranged units, hit air AND ground. Good vs minions, goblins and spear goblins.",
    "minions": "3 flying units. Ground troops with no ranged attack (giant, knight, mini-pekka) cannot hit them. "
               "Die to arrows, archers and musketeer.",
    "arrows": "spell, wide area, low damage. Kills minions, goblins and spear goblins. "
              "Only worth it on a group (2-3+ small units), aim at the group's centre.",
    "fireball": "spell, medium area, high damage. Kills musketeer/archers/spear goblins, can finish a "
                "low-HP tower. Do not waste it on a single cheap unit.",
}

# What the YOLO model can report for each enemy class name
UNIT_NOTES = {
    "giant": "tank that only targets buildings",
    "knight": "melee",
    "mini-pekka": "melee, very high damage",
    "musketeer": "ranged, hits air and ground",
    "archer": "ranged, hits air and ground (comes in pairs)",
    "minion": "flying swarm unit",
    "goblin": "fast melee swarm unit",
    "spr-goblin": "spear goblin, ranged swarm unit",
    "goblin-hut": "building that keeps spawning spear goblins",
    "arrows": "spell effect currently on the field",
    "fireball": "spell effect currently on the field",
}


def _build_system_prompt():
    deck = "\n".join(
        f"- {name} (cost {card['cost']}, {card['role']}): {CARD_TIPS.get(name, '')}"
        for name, card in CARDS.items()
    )
    units = "\n".join(f"- {name}: {UNIT_NOTES.get(name, UNIT_TYPES[name])}" for name in UNIT_TYPES)
    grid = "\n".join(f"- {line}" for line in legend())
    return f"""You are an expert Clash Royale player controlling the BOTTOM player ("user") in a live 1v1 match.
Every couple of seconds you get the current game state and decide ONE thing: play one card on one tile, or wait.

## How the game works
- Each side has 3 towers: a king tower in the middle at the back and two princess towers (left and right lane).
- Destroying the enemy king tower wins instantly. Otherwise, whoever destroyed more towers when time runs out wins.
- The king tower only starts attacking once it is hit or one of its princess towers is destroyed.
- Elixir goes from 0 to 10 and regenerates about 1 per 2.8 seconds. In the last 60 seconds it regenerates
  twice as fast. Elixir that would go above 10 is wasted, so do not sit at 10 for long.
- You hold 4 cards (slots 1-4). Playing a card replaces it with "next card", and the played card goes to the
  back of the queue. You can only play a card whose cost is <= your current elixir.
- Troops walk forward on their own and attack the closest target (giant only targets buildings).
  Your troops walk UP the screen (towards row 0), enemy troops walk DOWN (towards row {GAME_ROWS - 1}).
- The two lanes meet the river at the bridges; troops cross the river only at the bridges.

## The board
{grid}
- Enemy troops at row 15 or higher number (bridge / your side) are attacking you and need an answer.

## Where you can play
- TROOPS (anything that is not a spell) can only be placed on YOUR side: rows {OUR_FIRST_ROW}-{GAME_ROWS - 1},
  not on the river and not on top of a tower.
- SPELLS (arrows, fireball) can be placed on any tile, including the enemy side.
- Useful tiles: left lane column 3, right lane column 14.
  - Melee defender vs an attacker in that lane: (3, 21) or (14, 21), in front of your princess tower.
  - Ranged defender: (3, 27) or (14, 27), behind your princess tower so the tower tanks for it.
  - Middle (8, 20) / (9, 20): pulls enemy troops to the centre where both princess towers can shoot them.
  - Bridge push: (3, {OUR_FIRST_ROW}) or (14, {OUR_FIRST_ROW}).
  - Giant from the back: (8, {GAME_ROWS - 1}) for the left lane, (9, {GAME_ROWS - 1}) for the right lane.
  - Support a giant: 2-3 rows behind it (higher row number), same column.

## Your deck
{deck}

## Enemy units you may see
{units}

## Strategy
1. Defend first: if an enemy troop has reached the bridge or your side, answer it with the right counter
   in its lane (see the deck notes). Let your princess tower help - do not defend too far forward.
2. Win elixir trades: spend less elixir to answer a push than the enemy spent on it.
3. Spells only for value: arrows on a group of small units, fireball on musketeer/archers/spear goblins
   or to finish a low tower.
4. Attack: when elixir is high (8+) and nothing is threatening you, start a giant from the back, then add
   musketeer/archers/minions behind it as it walks. Push the lane where the enemy tower is weaker.
5. Do not dump all your elixir at once without a reason; keep enough to defend the counter-push.
6. Waiting is fine (and often correct) when nothing is happening and elixir is below ~8.
7. Positions come from computer vision and can be off by a tile or two; "?" means unknown.
   The troop list may contain the odd misdetection - do not overreact to a single strange unit.

## Answer format
Reply with ONLY one JSON object, no other text.
To play a card:
{{"action": "play", "card": "<card name exactly as written in your hand>", "tile": [column, row], "reason": "<max 15 words>"}}
To wait:
{{"action": "wait", "reason": "<max 15 words>"}}
column is 0-{GAME_COLUMNS - 1}, row is 0-{GAME_ROWS - 1}. Only pick a card that is in your hand and that you can afford now."""


SYSTEM_PROMPT = _build_system_prompt()


# ================= REPLY PARSING (never raises) =================
def parse_json_reply(text):
    """Model reply -> dict, or None. Copes with code fences / text around the JSON / Python-style quotes."""
    if not text:
        return None
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            try:
                data = ast.literal_eval(candidate)       # e.g. {'action': 'wait'}
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(data, dict):
            return data
    return None


def _normalise_card(name):
    """'Mini PEKKA' / 'mini-pekka' -> 'mini_pekka'."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _parse_tile(reply):
    """Accepts "tile": [c, r] / "(c, r)" / {"col": c, "row": r}, or top-level "col"/"row". Returns (c, r) or None."""
    tile = reply.get("tile", reply.get("position"))
    if isinstance(tile, str):
        tile = re.findall(r"-?\d+", tile)
    if isinstance(tile, dict):
        tile = [tile.get("col", tile.get("column", tile.get("x"))), tile.get("row", tile.get("y"))]
    if tile is None and "col" in reply and "row" in reply:
        tile = [reply["col"], reply["row"]]
    try:
        col, row = int(float(tile[0])), int(float(tile[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return col, row


def reply_to_action(reply, hand, elixir):
    """Parsed reply + CURRENT hand/elixir -> action dict, or None (wait / invalid).

    Returns the same format as RuleBasedBrain:
        {"slot": 2, "card": "arrows", "tile": (4, 18), "reason": "..."}
    """
    if not isinstance(reply, dict):
        return None
    action_type = str(reply.get("action", "play" if "card" in reply else "wait")).strip().lower()
    if action_type != "play":
        return None

    # Card: by name (preferred), or by slot number as a fallback
    card = _normalise_card(reply["card"]) if reply.get("card") is not None else None
    if card not in hand:
        try:
            slot = int(reply.get("slot"))
            card = hand[slot - 1] if 1 <= slot <= 4 else None
        except (TypeError, ValueError):
            card = None
    if card is None or card not in hand or card not in CARDS:
        return None
    if CARDS[card]["cost"] > elixir:
        return None

    tile = _parse_tile(reply)
    if tile is None:
        return None
    col = min(max(tile[0], 0), GAME_COLUMNS - 1)
    row = min(max(tile[1], 0), GAME_ROWS - 1)
    if CARDS[card]["role"] != "spell":
        row = max(row, OUR_FIRST_ROW)                 # troops can't go on the river / enemy side

    return {"slot": hand.index(card) + 1, "card": card, "tile": (col, row),
            "reason": str(reply.get("reason", ""))[:120]}


class LLMBrain:
    def __init__(self, model=MODEL, api_key=None):
        self.model = model
        self.client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT)   # api_key=None -> reads GROQ_API_KEY
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None               # Future of the request in flight
        self.last_request_time = None
        self.last_play_time = None
        self.paused_until = 0.0           # set after a rate-limit / API error
        self.recent_plays = deque(maxlen=RECENT_PLAYS)   # (time, card, tile)
        self.last_reply = None            # raw text of the last reply, for debugging
        self.last_error = None
        self.request_count = 0            # requests sent to the model so far
        self.last_request_seconds = None  # how long the last request took (reply or error)

    def decide(self, game_state, now=None):
        """Call often (e.g. every DECISION_INTERVAL). Never blocks on the network.

        game_state: GameState (its describe() is what the model reads)
        Returns None (wait / still thinking) or an action dict like RuleBasedBrain.decide().
        """
        now = time.monotonic() if now is None else now
        state = game_state.snapshot()
        action = None

        # 1. A reply has arrived -> check it against the state as it is NOW
        if self.pending is not None and self.pending.done():
            reply = self.pending.result()       # _ask() never raises
            self.pending = None
            in_cooldown = self.last_play_time is not None and now - self.last_play_time < PLAY_COOLDOWN
            if reply is not None and not in_cooldown:
                action = reply_to_action(reply, state["hand"], state["elixir"])
            if action is not None:
                self.last_play_time = now
                self.recent_plays.append((now, action["card"], action["tile"]))

        # 2. Nothing in flight -> ask again (unless throttled, or there's nothing we could play)
        if (self.pending is None and action is None
                and now >= self.paused_until
                and (self.last_request_time is None or now - self.last_request_time >= REQUEST_INTERVAL)
                and state["elixir"] >= MIN_CARD_COST):
            self.last_request_time = now
            self.request_count += 1
            self.pending = self.executor.submit(self._ask, self._user_message(game_state, state, now))

        return action

    def _user_message(self, game_state, state, now):
        affordable = [f"slot {slot} {card} ({CARDS[card]['cost']})"
                      for slot, card in enumerate(state["hand"], start=1)
                      if card in CARDS and CARDS[card]["cost"] <= state["elixir"]]
        plays = [f"{card} at {tile} {now - played:.0f}s ago" for played, card, tile in reversed(self.recent_plays)]
        return "\n".join([
            "Current game state:",
            *game_state.describe(),
            "",
            f"cards you can afford now: {', '.join(affordable) or 'none'}",
            f"your recent plays: {'; '.join(plays) or 'none'}",
            "",
            "Decide your move. Reply with the JSON object only.",
        ])

    def _ask(self, user_message):
        """Runs in the worker thread. Returns the parsed reply dict, or None on any failure."""
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                **model_options(self.model),
            )
            self.last_request_seconds = time.monotonic() - started
            self.last_reply = response.choices[0].message.content
            self.last_error = None
            return parse_json_reply(self.last_reply)
        except Exception as error:   # network, rate limit, bad request, ...: never take the bot down
            self.last_request_seconds = time.monotonic() - started
            self.last_error = f"{type(error).__name__}: {error}"
            # JSON mode rejects invalid JSON with a 400 that still carries the text -> try to salvage it
            body = getattr(error, "body", None)
            failed = body.get("error", {}).get("failed_generation") if isinstance(body, dict) else None
            if failed:
                self.last_reply = failed
                return parse_json_reply(failed)
            self.paused_until = time.monotonic() + RATE_LIMIT_BACKOFF
            return None
