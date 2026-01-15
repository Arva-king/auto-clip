import google.generativeai as genai
from google.api_core import exceptions

# Konfigurasi API Key
# NOTE: Jangan pernah menyebarkan script dengan API Key tertulis secara hardcode di environment produksi.
api_key = "AIzaSyBVE2bqNkvi4ygowSQqBVut_AiCn7N17yg"

try:
    genai.configure(api_key=api_key)

    print(f"Sedang mengambil daftar model dengan API Key: {api_key[:10]}... (disensor)\n")
    print("-" * 50)
    print(f"{'NAMA MODEL':<30} | {'METODE YANG DIDUKUNG'}")
    print("-" * 50)

    # Mengambil daftar model
    found_models = False
    for model in genai.list_models():
        found_models = True
        # Menampilkan nama model dan metode yang didukung (generateContent, embedContent, dll)
        methods = ", ".join(model.supported_generation_methods)
        print(f"{model.name:<30} | {methods}")

    if not found_models:
        print("Tidak ada model yang ditemukan. Cek akses API Key Anda.")

except exceptions.InvalidArgument:
    print("Error: API Key tidak valid.")
except exceptions.PermissionDenied:
    print("Error: Akses ditolak. Pastikan API Key memiliki izin yang benar.")
except Exception as e:
    print(f"Terjadi kesalahan tak terduga: {e}")

print("-" * 50)