"""
Backend API for Charco Chicken Syrian Restaurant
Handles order processing and storage
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uuid
import datetime
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Charco Chicken API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class OrderItem(BaseModel):
    name: str
    quantity: int
    price: float
    total: float

class OrderRequest(BaseModel):
    customer_name: str
    customer_phone: str
    items: List[Dict[str, Any]]
    total: float
    notes: Optional[str] = ""

class OrderResponse(BaseModel):
    order_id: str
    eta: int  # Estimated time in minutes
    status: str
    message: str

# In-memory storage (in production, use a real database)
orders_db = []
ORDER_FILE = "orders.json"

def load_orders():
    """Load orders from JSON file"""
    global orders_db
    if os.path.exists(ORDER_FILE):
        try:
            with open(ORDER_FILE, 'r', encoding='utf-8') as f:
                orders_db = json.load(f)
        except Exception as e:
            print(f"Error loading orders: {e}")
            orders_db = []
    else:
        orders_db = []

def save_orders():
    """Save orders to JSON file"""
    try:
        with open(ORDER_FILE, 'w', encoding='utf-8') as f:
            json.dump(orders_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving orders: {e}")

def calculate_eta(total_amount: float) -> int:
    """Calculate estimated delivery time based on order complexity"""
    base_time = 20  # Base 20 minutes
    
    # Add time based on order value
    if total_amount > 50:
        return base_time + 10
    elif total_amount > 30:
        return base_time + 5
    else:
        return base_time

@app.on_event("startup")
async def startup_event():
    """Load existing orders on startup"""
    load_orders()
    print("🚀 Charco Chicken API started!")
    print(f"📋 Loaded {len(orders_db)} existing orders")

@app.get("/")
async def root():
    return {
        "message": "مرحباً في API مطعم شاركو تشيكن! 🐔",
        "version": "1.0.0",
        "endpoints": {
            "submit_order": "POST /submit-order",
            "get_orders": "GET /orders",
            "get_order": "GET /orders/{order_id}",
            "menu": "GET /menu"
        }
    }

@app.post("/submit-order", response_model=OrderResponse)
async def submit_order(order: OrderRequest):
    """Submit a new order"""
    try:
        # Generate unique order ID
        order_id = f"CH{uuid.uuid4().hex[:6].upper()}"
        
        # Calculate ETA
        eta = calculate_eta(order.total)
        
        # Create order record
        order_record = {
            "order_id": order_id,
            "customer_name": order.customer_name,
            "customer_phone": order.customer_phone,
            "items": order.items,
            "total": order.total,
            "notes": order.notes,
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat(),
            "eta": eta,
            "estimated_delivery": (datetime.datetime.now() + datetime.timedelta(minutes=eta)).isoformat()
        }
        
        # Save to database
        orders_db.append(order_record)
        save_orders()
        
        return OrderResponse(
            order_id=order_id,
            eta=eta,
            status="confirmed",
            message=f"تم استلام طلبك! رقم الطلب: {order_id}. سيصل خلال {eta} دقيقة."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الطلب: {str(e)}")

@app.get("/orders")
async def get_all_orders():
    """Get all orders"""
    return {
        "total_orders": len(orders_db),
        "orders": orders_db
    }

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get specific order by ID"""
    for order in orders_db:
        if order["order_id"] == order_id:
            return order
    
    raise HTTPException(status_code=404, detail=f"الطلب رقم {order_id} غير موجود")

@app.get("/menu")
async def get_menu():
    """Get restaurant menu"""
    menu = {
        "مشاوي": [
            {"name": "فروج مشوي", "price": 25, "description": "فروج مشوي على الفحم مع الخضار"},
            {"name": "شاورما", "price": 15, "description": "شاورما دجاج أو لحمة"},
            {"name": "كباب", "price": 20, "description": "كباب مشكل مع الأرز"}
        ],
        "سلطات": [
            {"name": "فتوش", "price": 12, "description": "سلطة فتوش بالسماق والزعتر"},
            {"name": "تبولة", "price": 10, "description": "تبولة بالبقدونس والطماطم"}
        ],
        "مقبلات": [
            {"name": "حمص", "price": 8, "description": "حمص بالطحينة"},
            {"name": "متبل", "price": 8, "description": "متبل باذنجان"}
        ],
        "مشروبات": [
            {"name": "عصير ليمون", "price": 5, "description": "عصير ليمون طازج"},
            {"name": "شاي", "price": 3, "description": "شاي أحمر"},
            {"name": "قهوة", "price": 4, "description": "قهوة عربية"}
        ]
    }
    
    return {
        "restaurant": "شاركو تشيكن",
        "menu": menu,
        "currency": "ليرة سورية"
    }

@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    global orders_db
    
    for i, order in enumerate(orders_db):
        if order["order_id"] == order_id:
            if order["status"] == "pending":
                orders_db[i]["status"] = "cancelled"
                orders_db[i]["cancelled_at"] = datetime.datetime.now().isoformat()
                save_orders()
                return {"message": f"تم إلغاء الطلب رقم {order_id}"}
            else:
                raise HTTPException(status_code=400, detail="لا يمكن إلغاء الطلب في هذه المرحلة")
    
    raise HTTPException(status_code=404, detail=f"الطلب رقم {order_id} غير موجود")

@app.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, status: str):
    """Update order status"""
    valid_statuses = ["pending", "preparing", "ready", "delivered", "cancelled"]
    
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"حالة غير صحيحة. الحالات المتاحة: {valid_statuses}")
    
    for i, order in enumerate(orders_db):
        if order["order_id"] == order_id:
            orders_db[i]["status"] = status
            orders_db[i]["updated_at"] = datetime.datetime.now().isoformat()
            save_orders()
            return {"message": f"تم تحديث حالة الطلب {order_id} إلى {status}"}
    
    raise HTTPException(status_code=404, detail=f"الطلب رقم {order_id} غير موجود")

@app.get("/stats")
async def get_stats():
    """Get restaurant statistics"""
    if not orders_db:
        return {
            "total_orders": 0,
            "total_revenue": 0,
            "average_order_value": 0,
            "status_breakdown": {}
        }
    
    total_orders = len(orders_db)
    total_revenue = sum(order.get("total", 0) for order in orders_db)
    average_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    status_breakdown = {}
    for order in orders_db:
        status = order.get("status", "unknown")
        status_breakdown[status] = status_breakdown.get(status, 0) + 1
    
    return {
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "average_order_value": round(average_order_value, 2),
        "status_breakdown": status_breakdown,
        "currency": "ليرة سورية"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
