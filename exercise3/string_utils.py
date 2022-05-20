import string


def correct_string(str):
    str_ = str.split("-")

    if len(str_) == 1:

        if str_[0] == "s_GW":

            return "GW"

        elif str_[0] == "s_mi":

            return "-"

    str_corrected = [
        char[2:]
        if char
           in [
               "s_0",
               "s_1",
               "s_2",
               "s_3",
               "s_4",
               "s_5",
               "s_6",
               "s_7",
               "s_8",
               "s_9",
               "s_0th",
               "s_1st",
               "s_2nd",
               "s_3rd",
               "s_4th",
               "s_5th",
               "s_6th",
               "s_7th",
               "s_8th",
               "s_9th",
               "s_s",
           ]
        else char
        for char in str_
    ]

    # correct signature

    alphanumeric = list(string.ascii_letters) + list(string.digits)

    # each element is now potentially a substring, so check that each substring is alphanumeric
    str_without_specials = filter(
        lambda substr: all([char in alphanumeric for char in substr]), str_corrected
    )

    final_str = "".join(str_without_specials)

    return final_str
