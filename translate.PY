#dit is geen code maar een vertaling van de dataset had ik nodig voor de GUI input om  te  zetten  in een getal

cap_shape_translation = {
    "convex": 0,      # x=0
    "flat": 1,        # f=1
    "spherical": 2,   # p=2
    "bell": 3,        # b=3
    "conical": 4,     # c=4
    "sunken": 5,      # s=5
    "others": 6       # o=6
}

cap_surface_translation = {
    "grooves": 0,     # g=0
    "shiny": 1,       # h=1
    "sticky": 2,      # t=2
    "scaly": 3,       # y=3
    "fleshy": 4,      # e=4
    "smooth": 5,      # s=5
    "leathery": 6,    # l=6
    "dry": 7,         # d=7 ( d komt voor in de dataset  maar er is geen vertaling denk dat dit dry is ben niet zeker) 
    "wrinkled": 8,    # w=8
    "fibrous": 9,     # i=9
    "silky": 10       # k=10
}

cap_color_translation = {
    "orange": 0,   # o=0
    "red": 1,      # e=1
    "brown": 2,    # n=2
    "gray": 3,     # g=3
    "green": 4,    # r=4
    "white": 5,    # w=5
    "yellow": 6,   # y=6
    "pink": 7,     # p=7
    "purple": 8,   # u=8
    "buff": 9,     # b=9
    "blue": 10,    # l=10
    "black": 11    # k=11
}

does_bruise_or_bleed_translation = {
    "no": 0,       # f=0
    "yes": 1       # t=1
}

gill_attachment_translation = {
    "free": 0,        # e=0
    "adnate": 1,      # a=1
    "decurrent": 2,   # d=2
    "sinuate": 3,     # s=3
    "adnexed": 4,     # x=4
    "pores": 5,       # p=5
    "none": 6         # f=6
}

stem_root_translation = {
    "swollen": 0,     # s=0
    "bulbous": 1,     # b=1
    "rooted": 2,      # r=2
    "club": 3,        # c=3
    "Filamentous": 4      # f=4 → opniew komt voor in de data set maar geen vertaling gevonden (heb het op gezocht zou kunnen dat het Filamentous is)?
    # overige (cup, equal, rhizomorphs) komen niet voor in de dataset
}


has_ring_translation = {
    "ring": 0,    # t=0
    "none": 1     # f=1
}

ring_type_translation = {
    "grooved": 0,    # g=0
    "pendant": 1,    # p=1
    "evanescent": 2, # e=2
    "large": 3,      # l=3
    "none": 4,       # f=4
    "movable": 5,    # m=5
    "flaring": 6,    # r=6
    "zone": 7        # z=7
    # cobwebby, sheathing, scaly, komt niet voor in de dataset
}

habitat_translation = {
    "woods": 0,     # d=0
    "meadows": 1,   # m=1
    "grasses": 2,   # g=2
    "heaths": 3,    # h=3
    "leaves": 4,    # l=4
    "paths": 5,     # p=5
    "waste": 6,     # w=6
    "urban": 7      # u=7
}

season_translation = {
    "winter": 0,   # w=0
    "summer": 1,   # u=1
    "autumn": 2,   # a=2
    "spring": 3    # s=3
}
