import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from .database import AsyncSessionLocal
from .schema import User
from .auth import get_password_hash

async def create_superuser():
    print("--- AIRBOT Admin Creator ---")
    email = input("Admin Email: ").strip()
    name = input("Admin Name: ").strip()
    password = input("Password: ").strip()
    
    async with AsyncSessionLocal() as db:
        hashed = get_password_hash(password)
        new_admin = User(email=email, name=name, hashed_password=hashed, is_admin=True)
        db.add(new_admin)
        try:
            await db.commit()
            print(f"\n✅ Admin user '{email}' created successfully.")
        except Exception as e:
            print(f"\n❌ Failed to create user: {e}")

if __name__ == "__main__":
    asyncio.run(create_superuser())