from typing import Optional, Any

import pymongo
import uuid
from datetime import datetime

import config


class Database:
    def __init__(self):
        self.client = pymongo.MongoClient(config.mongodb_uri)
        self.db = self.client["chatgpt_telegram_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False, is_chat_id: bool = False):
        field_name = "chat_id" if is_chat_id else "_id"
        if self.user_collection.count_documents({field_name: user_id}) > 0:
            return True
        if raise_exception:
            raise ValueError(f"User {user_id} does not exist")
        return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,

            "username": username,
            "first_name": first_name,
            "last_name": last_name,

            "last_interaction": datetime.now(),
            "first_seen": datetime.now(),

            "current_dialog_id": None,
            "current_chat_mode": "assistant",
            "current_model": config.models["available_text_models"][2],

            "n_used_tokens": {},

            "n_generated_images": 0,
            "n_transcribed_seconds": 0.0  # voice message transcription
        }

        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)

    def start_new_dialog(self, user_id: int, is_chat_id=False):
        
        self.check_if_user_exists(user_id, raise_exception=True, is_chat_id=is_chat_id)
        
        dialog_id = str(uuid.uuid4())
        dialog_dict = {
            "_id": dialog_id,
            "user_id": user_id,
            "chat_mode": self.get_user_attribute(user_id, "current_chat_mode", is_chat_id=is_chat_id),
            "start_time": datetime.now(),
            "model": self.get_user_attribute(user_id, "current_model", is_chat_id=is_chat_id),
            "messages": []
        }

        # add new dialog
        self.dialog_collection.insert_one(dialog_dict)
        field_name = "chat_id" if is_chat_id else "_id"
        # update user's current dialog
        self.user_collection.update_one(
            {field_name: user_id},
            {"$set": {"current_dialog_id": dialog_id}}
        )

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str, is_chat_id=False):
        field_name = "chat_id" if is_chat_id else "_id"
        self.check_if_user_exists(user_id, raise_exception=True, is_chat_id=is_chat_id)
        user_dict = self.user_collection.find_one({field_name: user_id})

        if key not in user_dict:
            return None

        return user_dict[key]

    def set_user_attribute(self, user_id: int, key: str, value: Any, is_chat_id=False):
        field_name = "chat_id" if is_chat_id else "_id"
        self.check_if_user_exists(user_id, raise_exception=True, is_chat_id=is_chat_id)
        self.user_collection.update_one({field_name: user_id}, {"$set": {key: value}})

    def update_n_used_tokens(self, user_id: int, model: str, n_input_tokens: int, n_output_tokens: int, is_chat_id=False):
        n_used_tokens_dict = self.get_user_attribute(user_id, "n_used_tokens", is_chat_id=is_chat_id)

        if model in n_used_tokens_dict:
            n_used_tokens_dict[model]["n_input_tokens"] += n_input_tokens
            n_used_tokens_dict[model]["n_output_tokens"] += n_output_tokens
        else:
            n_used_tokens_dict[model] = {
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens
            }

        self.set_user_attribute(user_id, "n_used_tokens", n_used_tokens_dict, is_chat_id=is_chat_id)

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None, is_chat_id: bool = False):
        self.check_if_user_exists(user_id, raise_exception=True, is_chat_id=is_chat_id)
        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id", is_chat_id)

        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": user_id})
        if not dialog_dict:
            return []
        return dialog_dict["messages"]


    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None, is_chat_id: bool = False):
        self.check_if_user_exists(user_id, raise_exception=True, is_chat_id=is_chat_id)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id", is_chat_id)

        self.dialog_collection.update_one(
            {"_id": dialog_id, "user_id": user_id},
            {"$set": {"messages": dialog_messages}}
        )

