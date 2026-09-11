"""Checksum and format validators for common privacy identifiers."""

from datetime import datetime
import ipaddress
import re


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn_valid(value: str) -> bool:
    digits = digits_only(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        number = int(char)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def cn_id_card_valid(value: str) -> bool:
    normalized = value.strip().upper()
    if re.fullmatch(r"\d{15}", normalized):
        try:
            datetime.strptime("19" + normalized[6:12], "%Y%m%d")
        except ValueError:
            return False
        return normalized[:6] != "000000"
    if not re.fullmatch(r"\d{17}[\dX]", normalized):
        return False
    try:
        datetime.strptime(normalized[6:14], "%Y%m%d")
    except ValueError:
        return False
    if normalized[:6] == "000000":
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checksum = "10X98765432"
    total = sum(int(char) * weight for char, weight in zip(normalized[:17], weights))
    return checksum[total % 11] == normalized[-1]


USCC_ALPHABET = "0123456789ABCDEFGHJKLMNPQRTUWXY"
USCC_WEIGHTS = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)


def cn_uscc_valid(value: str) -> bool:
    normalized = value.strip().upper()
    if len(normalized) != 18 or any(char not in USCC_ALPHABET for char in normalized):
        return False
    total = sum(USCC_ALPHABET.index(char) * weight for char, weight in zip(normalized[:17], USCC_WEIGHTS))
    expected = USCC_ALPHABET[(31 - total % 31) % 31]
    return normalized[-1] == expected


def ipv4_valid(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def ipv6_valid(value: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(value.strip()), ipaddress.IPv6Address)
    except ValueError:
        return False


def mac_valid(value: str) -> bool:
    return re.fullmatch(r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}", value.strip()) is not None


def international_phone_valid(value: str) -> bool:
    digits = digits_only(value)
    return 7 <= len(digits) <= 15 and len(set(digits)) > 1


def card_expiry_valid(value: str) -> bool:
    return re.fullmatch(r"\s*(0?[1-9]|1[0-2])\s*[/.-]\s*(\d{2}|\d{4})\s*", value) is not None


def iban_valid(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for char in rearranged:
        encoded = char if char.isdigit() else str(ord(char) - ord("A") + 10)
        for digit in encoded:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1
