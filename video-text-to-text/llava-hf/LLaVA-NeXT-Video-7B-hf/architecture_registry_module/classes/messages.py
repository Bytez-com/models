from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ContentItem:
    type: str

    @property
    def value(self):
        pass


@dataclass
class ImageUrlContentItem(ContentItem):
    type: str
    image_url: str

    @property
    def value(self):
        return self.image_url


@dataclass
class TextContentItem(ContentItem):
    type: str
    text: str

    @property
    def value(self):
        return self.text


@dataclass
class Message:
    role: str
    content_items: List[ContentItem]
    pass


@dataclass
class Messages:
    items: List[Message]

    def from_json_list(messages: List[Dict]):
        items = []
        for message_dict in messages:
            content_items = []

            for item in message_dict["content"]:
                type = item["type"]

                if type == "text":
                    content_item = TextContentItem(type=type, text=item[type])
                    pass
                elif type == "image_url":
                    content_item = ImageUrlContentItem(type=type, image_url=item[type])
                    pass
                else:
                    raise Exception(
                        f'type: {type} is not a valid type, supported types: ["text", "image_url"]'
                    )

                content_items.append(content_item)

            message = Message(role=message_dict["role"], content_items=content_items)
            items.append(message)

        return Messages(items)
