#!/usr/bin/env python3
"""
tg_redial_dataset_loader.py

Load TG-ReDial dataset and produce JSONL samples in the format:
{
  "conversation_id": <int>,
  "message_id": <int>,
  "user_profile": [<str>, ...],
  "topics_discussed": [<str>, ...],
  "dialogue_history": [{"role": "User" | "Assistant", "content": <str>}, ...],  # up to this message
  "ground_truth_movie": <str|null>,
  "ground_truth_response": <str>
}
"""

import argparse
import pickle
import json
import csv
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Sample:
    conversation_id: int
    message_id: int
    user_profile: List[str]
    topics_discussed: List[str]
    dialogue_history: List[Dict[str, str]]  # only previous turns
    ground_truth_movie: Optional[str]
    ground_truth_response: str


class TGRedialDatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.conversations = []
        self.user_profiles = {}
        self.conv2user = None
        self.movie_id2name = {}

    def load(self, subset: str = "train"):
        """Load TG-ReDial dataset files"""
        conv_file = os.path.join(self.data_dir, f"{subset}_data.pkl")
        if not os.path.exists(conv_file):
            raise FileNotFoundError(f"{conv_file} not found")

        with open(conv_file, "rb") as f:
            self.conversations = pickle.load(f)
        print(f"Loaded {len(self.conversations)} conversations from {conv_file}")

        # User profiles
        user_profile_file = os.path.join(self.data_dir, "user2TopicSent.pkl")
        if os.path.exists(user_profile_file):
            with open(user_profile_file, "rb") as f:
                self.user_profiles = pickle.load(f)
            print(f"Loaded {len(self.user_profiles)} user profiles")
        else:
            print(f"Warning: {user_profile_file} not found.")

        # conv2user mapping
        conv2user_file = os.path.join(self.data_dir, "0619conv2user.pkl")
        if os.path.exists(conv2user_file):
            with open(conv2user_file, "rb") as f:
                self.conv2user = pickle.load(f)
            print("Loaded conv2user mapping")
        else:
            print("No conv2user mapping; using user_id from conversation")

        # movie ID -> name
        movie_file = os.path.join(self.data_dir, "movies_with_mentions.csv")
        if os.path.exists(movie_file):
            with open(movie_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        _, name_time, db_id = row[:3]
                        try:
                            db_id_int = int(db_id)
                            name = name_time.split('(')[0].strip()
                            self.movie_id2name[db_id_int] = name
                        except Exception:
                            continue
            print(f"Loaded {len(self.movie_id2name)} movie id->name mappings")

    def _get_user_profile(self, user_id: Any) -> List[str]:
        profile = self.user_profiles.get(str(user_id)) or self.user_profiles.get(user_id) or []
        if isinstance(profile, (set, tuple)):
            profile = list(profile)
        return [str(s) for s in profile]

    def extract_samples_from_conv(self, conv: Dict) -> List[Sample]:
        conv_id = conv.get("conv_id") or conv.get("conversation_id")
        messages = conv.get("messages", [])
        mention_movies = conv.get("mentionMovies", {})

        user_id = self.conv2user.get(str(conv_id)) if self.conv2user else conv.get("user_id")
        user_profile = self._get_user_profile(user_id)

        dialogue_history: List[Dict[str, str]] = []
        topics: List[str] = []
        samples: List[Sample] = []

        for msg in messages:
            message_id = int(msg.get("local_id", -1))
            role = msg.get("role")
            content = str(msg.get("content", ""))

            # Extract topics
            if "goal_path" in conv and message_id in conv["goal_path"]:
                goal_info = conv["goal_path"][message_id]
                if isinstance(goal_info, (list, tuple)) and len(goal_info) >= 3:
                    actions = goal_info[1::2]
                    keywords = goal_info[2::2]
                    for a, k in zip(actions, keywords):
                        if isinstance(k, str):
                            topics.append(k)
                        elif isinstance(k, (list, tuple)):
                            topics.extend([str(x) for x in k])

            # If it's a user message, add to dialogue history immediately
            if role == "Seeker":
                dialogue_history.append({"role": "User", "content": content})
            elif role == "Recommender" and message_id > 1:
                # create sample using dialogue history **up to this point**
                ground_truth_movie = None
                if message_id in mention_movies:
                    mov = mention_movies[message_id]
                    if isinstance(mov, (list, tuple)) and len(mov) >= 2:
                        mov_id, mov_name = mov[0], mov[1]
                        try:
                            mov_id_int = int(mov_id)
                            ground_truth_movie = self.movie_id2name.get(mov_id_int, mov_name)
                        except Exception:
                            ground_truth_movie = mov_name
                    else:
                        ground_truth_movie = str(mov)

                sample = Sample(
                    conversation_id=int(conv_id) if conv_id is not None else None,
                    message_id=message_id,
                    user_profile=user_profile.copy(),
                    topics_discussed=topics.copy(),
                    dialogue_history=dialogue_history.copy(),  # exclude this assistant response
                    ground_truth_movie=ground_truth_movie,
                    ground_truth_response=content
                )
                samples.append(sample)

                # finally append assistant turn to history AFTER creating sample
                dialogue_history.append({"role": "Assistant", "content": content})

        return samples

    def extract_all_samples(self) -> List[Sample]:
        all_samples: List[Sample] = []
        for conv in self.conversations:
            all_samples.extend(self.extract_samples_from_conv(conv))
        return all_samples


def save_samples_jsonl(samples: List[Sample], out_file: str):
    with open(out_file, "w", encoding="utf-8") as fout:
        for s in samples:
            obj = {
                "conversation_id": s.conversation_id,
                "message_id": s.message_id,
                "user_profile": s.user_profile,
                "topics_discussed": s.topics_discussed,
                "dialogue_history": s.dialogue_history,
                "ground_truth_movie": s.ground_truth_movie,
                "ground_truth_response": s.ground_truth_response
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="TG-ReDial -> JSONL dataset loader")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=str, default="tg_redial_prompts.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    loader = TGRedialDatasetLoader(args.data_dir)
    loader.load(args.subset)
    samples = loader.extract_all_samples()
    print(f"Total extracted samples: {len(samples)}")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    save_samples_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
