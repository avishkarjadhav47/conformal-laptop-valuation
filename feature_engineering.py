
def calculate_ppi(screen_size, resolution):
    width, height = (int(x) for x in resolution.split("x"))
    return (((width ** 2) + (height ** 2)) ** 0.5) / screen_size


def cpu_features(text):
    text = " ".join(str(text).split())

    if text.split()[0].isdigit():
        total_cores = int(text.split()[0])
    elif "Dual" in text:
        total_cores = 2
    elif "Quad" in text:
        total_cores = 4
    elif "Hexa" in text:
        total_cores = 6
    elif "Octa" in text:
        total_cores = 8
    else:
        total_cores = None

    threads = None
    if "Threads" in text:
        try:
            threads = int(text.split("Threads")[0].split()[-1])
        except ValueError:
            threads = None

    p_cores, e_cores = 0, 0
    if "(" in text and ")" in text:
        inside = text[text.find("(") + 1: text.find(")")]
        if "+" in inside:
            parts = inside.split("+")
            try:
                p_cores = int(parts[0].replace("P", "").strip())
                e_cores = int(parts[1].replace("E", "").strip())
            except ValueError:
                p_cores, e_cores = 0, 0

    hybrid_cpu = 1 if "(" in text else 0

    if "Dual" in text:
        core_type = "Dual"
    elif "Quad" in text:
        core_type = "Quad"
    elif "Hexa" in text:
        core_type = "Hexa"
    elif "Octa" in text:
        core_type = "Octa"
    else:
        core_type = "Numeric"

    if threads is not None and total_cores:
        hyperthreading = 1 if threads > total_cores else 0
        threads_per_core = threads / total_cores
    else:
        hyperthreading = 0
        threads_per_core = None

    return (total_cores, threads, p_cores, e_cores, hybrid_cpu, core_type, hyperthreading, threads_per_core)


def processor_features(text):
    text = " ".join(str(text).split())

    if "Intel" in text or "intel" in text:
        brand = "Intel"
    elif "AMD" in text or "Amd" in text:
        brand = "AMD"
    elif "Apple" in text:
        brand = "Apple"
    elif "MediaTek" in text:
        brand = "MediaTek"
    else:
        brand = "Other"

    generation = None
    if "Gen" in text:
        try:
            gen_text = text.split("Gen")[0].split()[-1]
            gen_digits = "".join(ch for ch in gen_text if ch.isdigit())
            generation = int(gen_digits) if gen_digits else None
        except (ValueError, IndexError):
            generation = None

    family_map = [
        ("Core i3", "Core i3"), ("Core i5", "Core i5"), ("Core i7", "Core i7"), ("Core i9", "Core i9"),
        ("Ryzen 3", "Ryzen 3"), ("Ryzen 5", "Ryzen 5"), ("Ryzen 7", "Ryzen 7"), ("Ryzen 9", "Ryzen 9"),
        ("Athlon", "Athlon"), ("Celeron", "Celeron"), ("Pentium", "Pentium"),
        ("Apple M2 Max", "Apple M2 Max"), ("Apple M2 Pro", "Apple M2 Pro"), ("Apple M2", "Apple M2"),
        ("Apple M1 Pro", "Apple M1 Pro"), ("Apple M1 Max", "Apple M1 Max"), ("Apple M1", "Apple M1"),
        ("MediaTek", "MediaTek"),
    ]
    family = "Other"
    for needle, label in family_map:
        if needle in text:
            family = label
            break

    return brand, family, generation


def gpu_features(text):
    text = " ".join(str(text).split())

    vram = 0
    if "GB" in text:
        temp = text.split("GB")[0].strip()
        if temp.isdigit():
            vram = int(temp)

    if "NVIDIA" in text or "Geforce" in text or "GeForce" in text or "Nvidia" in text:
        brand = "NVIDIA"
    elif "AMD" in text or "Radeon" in text:
        brand = "AMD"
    elif "Intel" in text or "Iris" in text or "UHD" in text or "HD Graphics" in text:
        brand = "Intel"
    elif "Apple" in text or "Core GPU" in text or "-core GPU" in text:
        brand = "Apple"
    elif "ARM" in text or "Mali" in text:
        brand = "ARM"
    else:
        brand = "Other"

    if "RTX" in text:
        family = "RTX"
    elif "GTX" in text:
        family = "GTX"
    elif "MX" in text:
        family = "MX"
    elif "Arc" in text:
        family = "Arc"
    elif "Quadro" in text:
        family = "Quadro"
    elif "Radeon" in text:
        family = "Radeon"
    elif "Iris Xe" in text or "Iris XE" in text or "Iris X" in text:
        family = "Iris Xe"
    elif "Iris Plus" in text:
        family = "Iris Plus"
    elif "UHD" in text:
        family = "UHD"
    elif "HD Graphics" in text or text == "Intel HD":
        family = "HD"
    elif "Mali" in text:
        family = "Mali"
    elif "-core GPU" in text:
        family = "Apple GPU"
    else:
        family = "Other"

    if brand in ["Intel", "Apple", "ARM"]:
        dedicated, integrated = 0, 1
    elif "Integrated" in text:
        dedicated, integrated = 0, 1
    else:
        dedicated, integrated = 1, 0

    return brand, family, vram, integrated, dedicated


def os_family(text):
    text = str(text)
    if "Windows" in text:
        return "Windows"
    elif "Mac" in text or "macOS" in text:
        return "macOS"
    elif "Ubuntu" in text:
        return "Linux"
    elif "Chrome" in text:
        return "Chrome"
    elif "Android" in text:
        return "Android"
    elif "DOS" in text:
        return "DOS"
    else:
        return "Other"
