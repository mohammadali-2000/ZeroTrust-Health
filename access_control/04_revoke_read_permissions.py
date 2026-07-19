import argparse
import asyncio
import py_nillion_client as nillion
import os
import sys
import pytest

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mpc_network_utils.nillion_client_helper import create_nillion_client
from mpc_network_utils.nillion_keypath_helper import getUserKeyFromFile, getNodeKeyFromFile

load_dotenv()

async def main(args = None):
    parser = argparse.ArgumentParser(
        description="Revoke user read/retrieve access_control from a secret on the Nillion network"
    )
    parser.add_argument(
        "--store_id",
        required=True,
        type=str,
        help="Store ID from the writer client store operation",
    )
    parser.add_argument(
        "--revoked_user_id",
        required=True,
        type=str,
        help="User ID of the reader python client (derived from user key)",
    )
    args = parser.parse_args(args)

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_2"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_4"))

    # Writer Nillion client
    writer = create_nillion_client(userkey, nodekey)

    # Create new access_control object to rewrite access_control (reader no longer has retrieve permission)
    new_access_control = nillion.Permissions.default_for_user(writer.user_id())
    result = (
        "allowed"
        if new_permissions.is_retrieve_allowed(args.revoked_user_id)
        else "not allowed"
    )
    if result != "not allowed":
        raise Exception("failed to create valid access_control object")

    # Update the permission
    print(f"ℹ️ Updating access_control for secret: {args.store_id}.") 
    print(f"ℹ️ Reset access_control so that user id {args.revoked_user_id} is {result} to retrieve object.", file=sys.stderr)
    await writer.update_permissions( cluster_id, args.store_id , new_permissions)

    print("\n\nRun the following command to test that access_control have been properly revoked")
    print(f"\n📋  python3 05_test_revoked_permissions.py  --store_id {args.store_id}")
    return args.store_id

if __name__ == "__main__":
    asyncio.run(main())

@pytest.mark.asyncio
async def test_main():
    pass
