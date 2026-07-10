from datetime import datetime
from typing import List

from pydantic import BaseModel


class Paper(BaseModel):

    title: str

    authors: List[str]

    summary: str

    published: datetime

    pdf_url: str

    entry_id: str

    categories: List[str]