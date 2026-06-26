from tortoise import Tortoise

from app.config import get_settings


TORTOISE_MODELS = ["app.models"]


async def init_db() -> None:
    settings = get_settings()
    await Tortoise.init(
        db_url=settings.database_url,
        modules={"models": TORTOISE_MODELS},
    )
    await Tortoise.generate_schemas(safe=True)


async def close_db() -> None:
    await Tortoise.close_connections()
