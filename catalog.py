import urllib.request
import lss
import tqdm

HYPERLEDA_URL="http://leda.univ-lyon1.fr"
HYPERLEDA_QUERY = HYPERLEDA_URL + "/fG.cgi?n={}&c=o&o={}&a=t"

def get_hyperleda_distance_mod(obj_name):
    if obj_name[0] in "0123456789":
        # Probably just a position name, change it so hyperleda can understand it:
        obj_name = "J" + obj_name

    query = HYPERLEDA_QUERY.format('a000', obj_name)

    response = urllib.request.urlopen(query)

    for line in response:
        line = line.decode()
        if line[0] == "#":
            continue
        #fields = line.split()
        #return float(fields[5]) # TODO

        # Dumb and hardcoded but whatever this is important if there are empty fields before the distance
        try:
            modc = float(line[66:71])
            return modc
        except:
            return None

def main():
    name = "MESSIER_088"
    print(get_hyperleda_distance_mod(name))

    catalog = lss.parse_catalog_data_mrs_pure("MRS/catalog/2mrs_1175_done.dat")
    
    catalog_filtered = [c for c in catalog if abs(c["v"]) < 2000]

    for c in tqdm.tqdm(catalog_filtered):
        d = get_hyperleda_distance_mod(c["name"])
        c["d"] = d

    print(catalog_filtered)

if __name__ == "__main__":
    main()