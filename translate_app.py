from deep_translator import GoogleTranslator

def tampilkan_header():
    """Menampilkan header ASCII art"""
    print("=" * 60)
    print("""
 ████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ████████╗███████╗
 ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗╚══██╔══╝██╔════╝
    ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║   ██║   █████╗  
    ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║   ██║   ██╔══╝  
    ██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║   ██║   ███████╗
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
    """)
    print("=" * 60)
    print("        Kelompok 5 - Project Capstone Bu Erna")
    print("=" * 60)
    print()

def tampilkan_anggota():
    """Menampilkan daftar anggota kelompok"""
    print("Anggota Kelompok:")
    print("-" * 60)
    print("1. Nama Anggota 1 (NPM: 123456789)")
    print("2. Nama Anggota 2 (NPM: 123456790)")
    print("3. Nama Anggota 3 (NPM: 123456791)")
    print("4. Nama Anggota 4 (NPM: 123456792)")
    print("-" * 60)
    print()

def translate_id_ke_en(teks):
    """Fungsi untuk translate dari Indonesia ke English"""
    try:
        translator = GoogleTranslator(source='id', target='en')
        hasil = translator.translate(teks)
        return hasil
    except Exception as e:
        return f"Error: {e}"

def main():
    """Fungsi utama program"""
    tampilkan_header()
    tampilkan_anggota()
    
    print("Program Translate Indonesia → English")
    print("=" * 60)
    print()
    
    while True:
        print("Masukkan teks dalam bahasa Indonesia (atau ketik 'exit' untuk keluar):")
        teks_input = input(">> ")
        
        if teks_input.lower() == 'exit':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        
        if teks_input.strip() == "":
            print("Teks tidak boleh kosong!\n")
            continue
        
        print("\nMenerjemahkan...")
        hasil_translate = translate_id_ke_en(teks_input)
        
        print("-" * 60)
        print(f"Teks Asli (ID): {teks_input}")
        print(f"Hasil (EN)    : {hasil_translate}")
        print("-" * 60)
        print()

if __name__ == "__main__":
    main()
