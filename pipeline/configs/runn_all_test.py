import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def remove_cache(folder_path):
    """Rimuove cache di pytest e __pycache__ se presenti"""
    for cache_dir in [".pytest_cache", "__pycache__"]:
        path = os.path.join(folder_path, cache_dir)
        if os.path.isdir(path):
            shutil.rmtree(path)



def run_test(folder_path, timeout_seconds=10):
    """Esegue pytest in una cartella e ritorna il risultato"""
    test_file = os.path.join(folder_path, "test_solution.py")
    result_file = os.path.join(folder_path, "test_result.txt")
    folder_name = os.path.basename(folder_path)

    if not os.path.isfile(test_file):
        return f"[!] Nessun test_solution.py trovato in {folder_name}, salto."

    remove_cache(folder_path)

    try:
        with open(result_file, "w") as f:
            process = subprocess.run(
                ["pytest", "test_solution.py", "-v"],
                cwd=folder_path,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds
            )
        if process.returncode == 0:
            return f"[✓] Test OK in {folder_name}"
        else:
            return f"[✗] Test FALLITI in {folder_name} (vedi test_result.txt)"
    except subprocess.TimeoutExpired:
        with open(result_file, "w") as f:
            f.write(f"[TIMEOUT] Superato limite di {timeout_seconds} secondi.\n")
        return f"TIMEOUT in {folder_name}"



def run_all_tests_parallel(base_dir, timeout_seconds=10, max_workers=8):
    folders = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()
    ], key=lambda x: int(os.path.basename(x)))

    print(f"==> Avvio test paralleli su {len(folders)} cartelle con {max_workers} thread\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_test, folder, timeout_seconds): folder
            for folder in folders
        }
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    base_directory = "./"  # Modifica se necessario
    timeout = 10  # Timeout in secondi
    workers = 4   # Numero di test in parallelo
    run_all_tests_parallel(base_directory, timeout_seconds=timeout, max_workers=workers)
