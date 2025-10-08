
#!/usr/bin/env python3
import os
import sys

API_KEY_FILE = os.path.join(os.path.dirname(__file__), 'api_key.py')

def main():
	if os.path.exists(API_KEY_FILE):
		print(f"'{API_KEY_FILE}' already exists. No changes made.")
		sys.exit(0)
	api_key = input("Enter your MapTiler API key: ").strip()
	if not api_key:
		print("No API key entered. Exiting.")
		sys.exit(1)
	with open(API_KEY_FILE, 'w') as f:
		f.write(f'API_KEY = "{api_key}"\n')
	print(f"API key saved to '{API_KEY_FILE}'.")

if __name__ == "__main__":
	main()
