from pydantic import BaseModel
from typing import Optional

class InventoryItem(BaseModel):
    Product_ID: Optional[str]
    Category: Optional[str]
    Region: Optional[str]
    Inventory_Level: Optional[float]
    Units_Sold: Optional[float]
    Deadstock: Optional[bool]
    Text_Feature: Optional[str]

class Recommendation(BaseModel):
    Product_ID: Optional[str]
    Category: Optional[str]
    Region: Optional[str]
    Distance: float
    Deadstock: bool
