import os
import sys
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Step 1: Initialize KiteConnect and get login URL
kite_api_key = os.getenv('KITE_API_KEY')
kite_api_secret = os.getenv('KITE_API_SECRET')
if not kite_api_key or not kite_api_secret:
    print("KITE_API_KEY or KITE_API_SECRET not found in .env")
    sys.exit(1)

kite = KiteConnect(api_key=kite_api_key)
kite.reqsession = __import__('requests').Session()
kite.reqsession.verify = False  # Disable SSL verification

print("Login URL:", kite.login_url())
print("Please login and provide the request token.")

# Step 2: Generate session with request token and API secret
request_token = input("Enter request token: ")
data = kite.generate_session(request_token, kite_api_secret)
access_token = data["access_token"]

# Save access token to .env
env_path = '.env'
set_key(env_path, 'KITE_ACCESS_TOKEN', access_token)
print("Access token saved to .env as KITE_ACCESS_TOKEN")