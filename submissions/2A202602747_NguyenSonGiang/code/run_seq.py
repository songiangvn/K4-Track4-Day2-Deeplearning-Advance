"""run_seq.py - chạy lần lượt nhiều script trong một job SLURM; một lệnh lỗi không chặn các lệnh sau.

    python run_seq.py "method_select.py --runs C01s/seed0 --tag MS_C01s" "res_sweep.py --pattern B*/seed0"

Mỗi tham số là một lệnh (tách bằng shlex), chạy bằng cùng trình thông dịch Python. Thoát với mã 1 nếu có
lệnh lỗi, và in tóm tắt cuối cùng.
"""
import shlex
import subprocess
import sys


def main() -> int:
    results = []
    for cmd in sys.argv[1:]:
        args = shlex.split(cmd)
        print(f"\n===== {cmd}", flush=True)
        code = subprocess.run([sys.executable, *args]).returncode
        results.append((cmd, code))
    print("\n===== TÓM TẮT")
    for cmd, code in results:
        print(("XONG " if code == 0 else f"LỖI ({code}) ") + cmd, flush=True)
    return int(any(code != 0 for _, code in results))


if __name__ == "__main__":
    sys.exit(main())
