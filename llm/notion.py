from typing import Optional
from notion_client import Client
import datetime

class Notion:
    def __init__(self, token: str, parent_page_id: Optional[str] = None, notebook_title: str | None = None):
        try:
            self.notion = Client(auth=token)
            me = self.notion.users.me()   # lightweight test call
            print("Connected as:", me["name"])
            self.connected = True
        except Exception as e:
            print("Notion connection failed:", e)
            self.connected = False
            return

        # Create a unique notebook title if none provided
        if notebook_title is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notebook_title = f"Notebook - {timestamp}"

        # Create a new page/notebook
        response = self.notion.pages.create(
            parent = {"page_id": parent_page_id},
            properties={
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {
                                "content": notebook_title
                            }
                        }
                    ]
                }
            }
        )

        self.page_id = response["id"]
        
        
    def is_connected(self) -> bool:
        return self.is_connected
        

    def write(self, text: str):
        self.notion.blocks.children.append(
            block_id=self.page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}
            }]
        )
        
        
    def write_blocks(self, blocks: list):
        self.notion.blocks.children.append(
            block_id=self.page_id,
            children=blocks
        )
       
        
    def conversation_to_notion_blocks(self, conversation_history):
        """
        Convert a list of {"role": ..., "content": ...} into Notion block payloads.
        - User messages → heading 3 (mini paragraph titles)
        - Assistant messages → normal paragraph text
        """
        blocks = []
        
        for turn in conversation_history:
            if turn["role"] == "user":
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": turn["content"]}}]
                    }
                })
                
            elif turn["role"] == "assistant":
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": turn["content"]}}]
                    }
                })
                
        return blocks