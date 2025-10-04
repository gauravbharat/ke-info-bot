import asyncio
from langchain_community.document_loaders import JSONLoader


async def ingest():
    # 1. Load JSON files data each with json file name and url based on the file name

    # 2. Split data into separate, semantically meaningful chunks

    # 3. Embed data, create a vector space

    # 4. Store data in vector database
    pass


if __name__ == "__main__":
    asyncio.run(ingest())
