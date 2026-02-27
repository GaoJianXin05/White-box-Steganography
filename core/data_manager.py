import os


class BitStreamReader:

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.bits = f.read().strip()
        self.pos = 0
        self.total_embedded = 0

    def read_bits(self, n: int) -> str:
        result = []
        for _ in range(n):
            result.append(self.bits[self.pos % len(self.bits)])
            self.pos += 1
        self.total_embedded += n
        return "".join(result)

    def peek_bits(self, n: int) -> str:
        result = []
        for i in range(n):
            result.append(self.bits[(self.pos + i) % len(self.bits)])
        return "".join(result)

    def snapshot(self):
        return (self.pos, self.total_embedded)

    def restore(self, snap):
        self.pos, self.total_embedded = snap


class StegoDataManager:

    @staticmethod
    def method_dir(output_dir: str, method: str) -> str:
        return os.path.join(output_dir, method)

    @staticmethod
    def context_file(output_dir: str, method: str) -> str:
        return os.path.join(output_dir, method, "contexts.txt")

    @staticmethod
    def cover_file(output_dir: str, method: str) -> str:
        return os.path.join(output_dir, method, "cover.txt")

    @staticmethod
    def stego_file(output_dir: str, method: str) -> str:
        return os.path.join(output_dir, method, "stego.txt")

    @staticmethod
    def stego_bits_file(output_dir: str, method: str) -> str:
        return os.path.join(output_dir, method, "stego_bits.txt")

    @staticmethod
    def load_contexts(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def normalize_count(items: list, n: int) -> list:
        if len(items) == 0:
            raise ValueError("Empty list, cannot normalize.")
        if len(items) >= n:
            return items[:n]
        result = []
        while len(result) < n:
            result.extend(items)
        return result[:n]

    @staticmethod
    def save_cover_result(cover_texts: list, output_dir: str, method: str):
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        path = os.path.join(method_dir, "cover.txt")
        with open(path, "w", encoding="utf-8") as f:
            for t in cover_texts:
                f.write(t + "\n")
        print(f"  Cover saved: {path}  ({len(cover_texts)} lines)")

    @staticmethod
    def save_stego_result(method: str, stego_texts: list, stego_bits: list, output_dir: str):
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        stego_path = os.path.join(method_dir, "stego.txt")
        with open(stego_path, "w", encoding="utf-8") as f:
            for t in stego_texts:
                f.write(t + "\n")

        bits_path = os.path.join(method_dir, "stego_bits.txt")
        with open(bits_path, "w", encoding="utf-8") as f:
            for b in stego_bits:
                f.write(b + "\n")

        print(f"  Stego saved: {stego_path}  ({len(stego_texts)} lines)")
        print(f"  Bits  saved: {bits_path}")