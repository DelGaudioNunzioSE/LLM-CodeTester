import os
import subprocess
import shutil

# === Percorsi principali ===
TEST_FOLDER = os.path.abspath(os.path.join("..", "DatasetTESTGEN"))
OUTPUT_FOLDER = os.path.abspath(os.path.join("..", "DatasetTESTGENOUTPUT"))

# === Crea cartella output se non esiste ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

executed = 0
skipped = 0
copied = 0

print(f"\n=== Running tests from: {TEST_FOLDER} ===\n")

for folder_name in os.listdir(TEST_FOLDER):
    folder_path = os.path.join(TEST_FOLDER, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"--- Processing: {folder_name} ---")

    run_file = os.path.join(folder_path, "run_test.bat")
    if not os.path.isfile(run_file):
        print(f"[WARN] {folder_name} → no run_test.bat found.")
        skipped += 1
        continue

    result_file = os.path.join(folder_path, "test_result.txt")

    if not os.path.exists(result_file):
        print(f"[RUN ] {folder_name}")
        try:
            subprocess.run(run_file, cwd=folder_path, shell=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {folder_name} → test took too long.")
    else:
        print(f"[SKIP] {folder_name} → test_result.txt already exists.")
        skipped += 1

    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
            result = f.read().strip()
        print(f"[RESULT] {folder_name} → {result}")
        if "fail" in result.lower() or "no result" in result.lower():
            dest = os.path.join(OUTPUT_FOLDER, folder_name)
            shutil.copytree(folder_path, dest, dirs_exist_ok=True)
            print(f"[COPY] {folder_name} → copied to OUTPUT.")
            copied += 1
    else:
        print(f"[FAIL] {folder_name} → test_result.txt not found.")
        dest = os.path.join(OUTPUT_FOLDER, folder_name)
        shutil.copytree(folder_path, dest, dirs_exist_ok=True)
        print(f"[COPY] {folder_name} → copied to OUTPUT (no result).")
        copied += 1

    executed += 1
    print()

# === Report finale ===
print("=== SUMMARY ===")
print(f"Executed: {executed}")
print(f"Skipped : {skipped}")
print(f"Copied  : {copied}")
