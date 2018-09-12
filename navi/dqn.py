import json
import psycopg2
import pandas as pds
import keras.backend as K
import numpy as np
import tensorflow as tf
from collections import deque
import configparser

graph = []
GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.stateData = pds.DataFrame()
        self.combatDataDict = {}
        self.stateDataDict = {}
        self.current_state = pds.DataFrame()
        self.previous_state = pds.DataFrame()
        self.current_state_array = np.zeros((1, 5951))
        self.previous_state_array = np.zeros((1, 5951))
        self.current_action = np.zeros(21)
        self.previous_action = np.zeros(21)
        self.action_list = []
        #self.model = self.create_model()
        self.turn = 1

    def loadStateDataToDatabase(self):
        data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\StateDataDumpjsonDump.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        self.loadDataToDatabase(data, "stateData")

    def loadCombatDataToDatabase(self):
        data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\CombatDataDumpjsonDump.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        i=0
        for enemy in data["jsonEnemyArrayList"]:
            corrected_dict = {(k + str(i)): v for k, v in enemy.items()}
            data.update(corrected_dict)
            i=i+1
        del data["jsonEnemyArrayList"]

        j=0
        for card in data["jsonCardArrayListHand"]:
                corrected_dict = {(k + str(j)): v for k, v in card.items()}
        data.update(corrected_dict)
        j=j+1
        del data["jsonCardArrayListHand"]

        l=0
        for card in data["jsonCardArrayListExhaustPile"]:
            corrected_dict = {(k + str(l)): v for k, v in card.items()}
        data.update(corrected_dict)
        l=l+1
        del data["jsonCardArrayListExhaustPile"]

        m=0
        for card in data["jsonCardArrayListDiscardPile"]:
            corrected_dict = {(k + str(m)): v for k, v in card.items()}
        data.update(corrected_dict)
        m=m+1
        del data["jsonCardArrayListDiscardPile"]

        n=0
        for card in data["jsonCardArrayListDrawPile"]:
            corrected_dict = {(k + str(n)): v for k, v in card.items()}
        data.update(corrected_dict)
        n=n+1
        del data["jsonCardArrayListDrawPile"]

        self.loadDataToDatabase(data, "combatStateData")

        config = configparser.ConfigParser()
        config.read('config.ini')
        #conn = psycopg2.connect(host=config['postgresql']['host'], database=config['postgresql']['database'],
        #                        user=config['postgresql']['user'], password=config['postgresql']['password'], port=config['postgresql']['port'])
        #print("Database Connected")

        #with conn.cursor() as cursor:
        #    keys = data.keys()
        #    columns = ','.join(keys)
        #    values = ','.join(["%("+k+")s" for k in data])
        #    insert = 'INSERT into {0} ({1}) VALUES ({2})'.format("combatStateData", columns, values)
        #    print(insert, data)
        #    cursor.execute(insert, data)
        #conn.commit()


    def loadDataToDatabase(self, data_dict, table_name):
        conn = psycopg2.connect(host="localhost", database="stsData",
                            user="postgres", password="postgres", port='5433')
        print("Database Connected")

        with conn.cursor() as cursor:
            keys = data_dict.keys()
            columns = ','.join(keys)
            values = ','.join(["%("+k+")s" for k in data_dict])
            insert = 'INSERT into {0} ({1}) VALUES ({2})'.format(table_name, columns, values)
            cursor.execute(insert, data_dict)
        conn.commit()

    def loadAllDataToDatabase(self):
        self.loadStateDataToDatabase()
        self.loadCombatDataToDatabase()

    @staticmethod
    def classNameToInt(className):
        return {
            'IRONCLAD': 0,
            'SILENT': 1,
            'DEFECT': 2
        }[className]

    @staticmethod
    def cardIdToInt(cardId):
        return {
                '': -1,
                'A Thousand Cuts': 0,
                'Accuracy': 1,
                'Acrobatics': 2,
                'Adrenaline': 3,
                'After Image': 4,
                'Aggregate': 5,
                'All For One': 6,
                'All Out Attack': 7,
                'Allocate': 8,
                'Amplify': 9,
                'Anger': 10,
                'Apotheosis': 11,
                'Armaments': 12,
                'AscendersBane': 13,
                'Auto Shields': 14,
                'Axe Kick': 15,
                'Backflip': 16,
                'Backstab': 17,
                'Ball Lightning': 18,
                'Bandage Up': 19,
                'Bane': 20,
                'Barrage': 21,
                'Barricade': 22,
                'Bash': 23,
                'Battle Trance': 24,
                'Beam Cell': 25,
                'Berserk': 26,
                'Biased Cognition': 27,
                'Bite': 28,
                'Blade Dance': 29,
                'Blaster': 30,
                'Blind': 31,
                'Blizzard': 32,
                'Blood for Blood': 33,
                'Bloodletting': 34,
                'Bludgeon': 35,
                'Blur': 36,
                'Body Slam': 37,
                'BootSequence': 38,
                'Bouncing Flask': 39,
                'Brutality': 40,
                'Buffer': 41,
                'Bullet Time': 42,
                'Burn': 43,
                'Burning Pact': 44,
                'Burst': 45,
                'Cache': 46,
                'Calculated Gamble': 47,
                'Caltrops': 48,
                'Capacitor': 49,
                'Carnage': 50,
                'Catalyst': 51,
                'Channel': 52,
                'Chaos': 53,
                'Chill': 54,
                'Choke': 55,
                'Chrysalis': 56,
                'Clash': 57,
                'Cleave': 58,
                'Cloak And Dagger': 59,
                'Clothesline': 60,
                'Clumsy': 61,
                'Cold Snap': 62,
                'Combust': 63,
                'Concentrate': 64,
                'Conserve Battery': 65,
                'Consume': 66,
                'Coolheaded': 67,
                'Corpse Explosion': 68,
                'Corruption': 69,
                'Creative AI': 70,
                'Crippling Poison': 71,
                'Dagger Spray': 72,
                'Dagger Throw': 73,
                'Dark Embrace': 74,
                'Dark Shackles': 75,
                'Darkness': 76,
                'Dash': 77,
                'Dazed': 78,
                'Deadly Poison': 79,
                'Decay': 80,
                'Deep Breath': 81,
                'Defend_B': 82,
                'Defend_G': 83,
                'Defend_R': 84,
                'Deflect': 85,
                'Defragment': 86,
                'Demon Form': 87,
                'Die Die Die': 88,
                'Disarm': 89,
                'Distraction': 90,
                'Dodge and Roll': 91,
                'Doom and Gloom': 92,
                'Doppelganger': 93,
                'Double Energy': 94,
                'Double Tap': 95,
                'Doubt': 96,
                'Dramatic Entrance': 97,
                'Dropkick': 98,
                'Dual Wield': 99,
                'Dualcast': 100,
                'Echo Form': 101,
                'Electrodynamics': 102,
                'Endless Agony': 103,
                'Energy Pulse': 104,
                'Enlightenment': 105,
                'Entrench': 106,
                'Envenom': 107,
                'Escape Plan': 108,
                'Eviscerate': 109,
                'Evolve': 110,
                'Exhume': 111,
                'Expertise': 112,
                'FTL': 113,
                'Feed': 114,
                'Feel No Pain': 115,
                'Fiend Fire': 116,
                'Finesse': 117,
                'Finisher': 118,
                'Fire Breathing': 119,
                'Fission': 120,
                'Flame Barrier': 121,
                'Flash of Steel': 122,
                'Flechettes': 123,
                'Flex': 124,
                'Flux Capacitor': 125,
                'Flying Knee': 126,
                'Footwork': 127,
                'Force Field': 128,
                'Fusion': 129,
                'Gash': 130,
                'Genetic Algorithm': 131,
                'Ghostly': 132,
                'Ghostly Armor': 133,
                'Glacier': 134,
                'Glass Knife': 135,
                'Go for the Eyes': 136,
                'Good Instincts': 137,
                'Grand Finale': 138,
                'Havoc': 139,
                'Headbutt': 140,
                'Heatsinks': 141,
                'Heavy Blade': 142,
                'Heel Hook': 143,
                'Hello World': 144,
                'Hemokinesis': 145,
                'Hide': 146,
                'Hologram': 147,
                'Hyperbeam': 148,
                'Ice Wall': 149,
                'Immolate': 150,
                'Impervious': 151,
                'Impulse': 152,
                'Infernal Blade': 153,
                'Infinite Blades': 154,
                'Inflame': 155,
                'Injury': 156,
                'Intimidate': 157,
                'Iron Wave': 158,
                'J.A.X.': 159,
                'Jack Of All Trades': 160,
                'Juggernaut': 161,
                'Leap': 162,
                'Leg Sweep': 163,
                'Limit Break': 164,
                'Lockon': 165,
                'Loop': 166,
                'Machine Learning': 167,
                'Madness': 168,
                'Magnetism': 169,
                'Malaise': 170,
                'Master of Strategy': 171,
                'Masterful Stab': 172,
                'Mayhem': 173,
                'Melter': 174,
                'Metallicize': 175,
                'Metamorphosis': 176,
                'Meteor Strike': 177,
                'Mind Blast': 178,
                'Multi-Cast': 179,
                'Necronomicurse': 180,
                'Neutralize': 181,
                'Night Terror': 182,
                'Normality': 183,
                'Nova': 184,
                'Noxious Fumes': 185,
                'Offering': 186,
                'Outmaneuver': 187,
                'Overclock': 188,
                'Pain': 189,
                'Panacea': 190,
                'Panache': 191,
                'Parasite': 192,
                'Perfected Strike': 193,
                'Phantasmal Killer': 194,
                'PiercingWail': 195,
                'Poisoned Stab': 196,
                'Pommel Strike': 197,
                'Power Through': 198,
                'Predator': 199,
                'Prepared': 200,
                'Prime': 201,
                'Primitive Tech': 202,
                'Pummel': 203,
                'Purity': 204,
                'Quick Slash': 205,
                'Rage': 206,
                'Rainbow': 207,
                'Rampage': 208,
                'Reaper': 209,
                'Reboot': 210,
                'Rebound': 211,
                'Reckless Charge': 212,
                'Recycle': 213,
                'Redo': 214,
                'Reflex': 215,
                'Regret': 216,
                'Reinforced Body': 217,
                'Reprieve': 218,
                'Reprogram': 219,
                'Riddle With Holes': 220,
                'Rip and Tear': 221,
                'Rupture': 222,
                'Sadistic Nature': 223,
                'Scrape': 224,
                'Searing Blow': 225,
                'Second Wind': 226,
                'Secret Technique': 227,
                'Secret Weapon': 228,
                'Seeing Red': 229,
                'Seek': 230,
                'Self Repair': 231,
                'Sentinel': 232,
                'Setup': 233,
                'Sever Soul': 234,
                'Shame': 235,
                'Shiv': 236,
                'Shockwave': 237,
                'Shrug It Off': 238,
                'Skewer': 239,
                'Skim': 240,
                'Slice': 241,
                'Slimed': 242,
                'Spot Weakness': 243,
                'Stack': 244,
                'Static Discharge': 245,
                'Steam': 246,
                'Steam Power': 247,
                'Storm': 248,
                'Storm of Steel': 249,
                'Streamline': 250,
                'Strike_B': 251,
                'Strike_G': 252,
                'Strike_R': 253,
                'Sucker Punch': 254,
                'Sunder': 255,
                'Survivor': 256,
                'Sweeping Beam': 257,
                'Swift Strike': 258,
                'Sword Boomerang': 259,
                'Tactician': 260,
                'Tempest': 261,
                'Terror': 262,
                'The Bomb': 263,
                'Thinking Ahead': 264,
                'Thunder Strike': 265,
                'Thunderclap': 266,
                'Tools of the Trade': 267,
                'Transmutation': 268,
                'Trip': 269,
                'True Grit': 270,
                'Turbo': 271,
                'Twin Strike': 272,
                'Underhanded Strike': 273,
                'Undo': 274,
                'Unload': 275,
                'Uppercut': 276,
                'Venomology': 277,
                'Void': 278,
                'Warcry': 279,
                'Well Laid Plans': 280,
                'Whirlwind': 281,
                'White Noise': 282,
                'Wild Strike': 283,
                'Winter': 284,
                'Wound': 285,
                'Wraith Form v2': 286,
                'Writhe': 287,
                'Zap': 288,
                'Core Surge': 289,
                'Compile Driver': 290,
                'RitualDagger': 291,
                'Pride': 292,
                'PanicButton': 293,
                'HandOfGreed': 294,
                'Violence': 295,
                'Impatience': 296,
                'Forethought': 297,
                'Discovery': 298
                }[cardId]

    @staticmethod
    def powerIdToInt(powerId):
        return {
            '': -1,
            'Accuracy': 0,
            'After Image': 1,
            'Amplify': 2,
            'Anger': 3,
            'Angry': 4,
            'Artifact': 5,
            'Attack Burn': 6,
            'Barricade': 7,
            'Berserk': 8,
            'Bias': 9,
            'Blur': 10,
            'Brutality': 11,
            'Buffer': 12,
            'Bullet Time': 13,
            'Burst': 14,
            'Choked': 15,
            'Combust': 16,
            'Confusion': 17,
            'Conserve': 18,
            'Constricted': 19,
            'Corruption': 20,
            'Creative AI': 21,
            'Curiosity': 22,
            'Curl Up': 23,
            'Dance Puppet': 24,
            'Dark Embrace': 25,
            'Demon Form': 26,
            'DexLoss': 27,
            'Dexterity': 28,
            'Double Damage': 29,
            'Double Tap': 30,
            'Draw': 31,
            'Draw Card': 32,
            'Draw Down': 33,
            'Draw Reduction': 34,
            'Echo Form': 35,
            'Electro': 36,
            'Energized': 37,
            'EnergizedBlue': 38,
            'Entangled': 39,
            'Envenom': 40,
            'Evolve': 41,
            'Explosive': 42,
            'Extra Cards': 43,
            'Feel No Pain': 44,
            'Fire Breathing': 45,
            'Flame Barrier': 46,
            'Flex': 47,
            'Flight': 48,
            'Focus': 49,
            'Frail': 50,
            'Gambit': 51,
            'Generic Strength Up Power': 52,
            'GrowthPower': 53,
            'Heatsink': 54,
            'Hello': 55,
            'Hex': 56,
            'Hide': 57,
            'Infinite Blades': 58,
            'Intangible': 59,
            'IntangiblePlayer': 60,
            'Inverted': 61,
            'Juggernaut': 62,
            'Knowledge': 63,
            'Life Link': 64,
            'Lockon': 65,
            'Loop': 66,
            'Machine Learning': 67,
            'Magnetism': 68,
            'Malleable': 69,
            'Mayhem': 70,
            'Metallicize': 71,
            'Minion': 72,
            'Mode Shift': 73,
            'Next Turn Block': 74,
            'Night Terror': 75,
            'No Attack': 76,
            'No Draw': 77,
            'Noxious Fumes': 78,
            'Nullify Attack': 79,
            'Painful Stabs': 80,
            'Panache': 81,
            'Pen Nib': 82,
            'Phantasmal': 83,
            'Plated Armor': 84,
            'Poison': 85,
            'Prime': 86,
            'Primitive': 87,
            'Rage': 88,
            'Rebound': 89,
            'Reduce Damage': 90,
            'Regenerate': 91,
            'Regeneration': 92,
            'Repair': 93,
            'Repulse': 94,
            'Retain Cards': 95,
            'Retain Hand': 96,
            'Riposte': 97,
            'Ritual': 98,
            'Rupture': 99,
            'Sadistic': 100,
            'Serpentine': 101,
            'Shackled': 102,
            'Sharp Hide': 103,
            'Shriek From Beyond': 104,
            'Skill Burn': 105,
            'Slow': 106,
            'Split': 107,
            'Spore Cloud': 108,
            'Stasis': 109,
            'StaticDischarge': 110,
            'Storm': 111,
            'Strength': 112,
            'TheBomb': 113,
            'Thievery': 114,
            'Thorns': 115,
            'Thousand Cuts': 116,
            'Time Warp': 117,
            'Tools Of The Trade': 118,
            'Unawakened': 119,
            'Unstable': 120,
            'Venomology': 121,
            'Vulnerable': 122,
            'Weakened': 123,
            'Wraith Form v2': 124,
            'Equilibrium': 125,
            'Fading': 126,
            'Shifting': 127,
            'TimeMazePower': 128,
            'NoBlockPower': 129,
            'AutoPlayFormPower': 130
            }[powerId]

    @staticmethod
    def relicIdToInt(relicId):
        return {
            '': -1,
            'Anchor': 0,
            'Ancient Tea Set': 1,
            'Art of War': 2,
            'Astrolabe': 3,
            'Bag of Marbles': 4,
            'Bag of Preparation': 5,
            'Bird Faced Urn': 6,
            'Black Blood': 7,
            'Black Star': 8,
            'Blood Vial': 9,
            'Bloody Idol': 10,
            'Blue Candle': 11,
            'Bottled Flame': 12,
            'Bottled Lightning': 13,
            'Bottled Tornado': 14,
            'Bronze Scales': 15,
            'Burning Blood': 16,
            'Cables': 17,
            'Calipers': 18,
            'Calling Bell': 19,
            'Cauldron': 20,
            'Centennial Puzzle': 21,
            'Chameleon Ring': 22,
            'Champion Belt': 23,
            'Charon\'s Ashes': 24,
            'Chemical X': 25,
            'Circlet': 26,
            'Cracked Core': 27,
            'FrozenCore': 28,
            'Cursed Key': 29,
            'Darkstone Periapt': 30,
            'DataDisk': 31,
            'Dead Branch': 32,
            'Derp Rock': 33,
            'Discerning Monocle': 34,
            'Dodecahedron': 35,
            'Dream Catcher': 36,
            'Du-Vu Doll': 37,
            'Ectoplasm': 38,
            'Emotion Chip': 39,
            'Enchiridion': 40,
            'Eternal Feather': 41,
            'Frozen Egg': 42,
            'Frozen Egg 2': 43,
            'Frozen Eye': 44,
            'Gambling Chip': 45,
            'Ginger': 46,
            'Girya': 47,
            'Golden Idol': 48,
            'Gremlin Horn': 49,
            'Happy Flower': 50,
            'Ice Cream': 51,
            'Inserter': 52,
            'Juzu Bracelet': 53,
            'Kunai': 54,
            'Lantern': 55,
            'Lee\'s Waffle': 56,
            'Letter Opener': 57,
            'Living Blade': 58,
            'Lizard Tail': 59,
            'Magic Flower': 60,
            'Mango': 61,
            'Mark of Pain': 62,
            'Matryoshka': 63,
            'Meat on the Bone': 64,
            'Medical Kit': 65,
            'Membership Card': 66,
            'Mercury Hourglass': 67,
            'Molten Egg': 68,
            'Molten Egg 2': 69,
            'Mummified Hand': 70,
            'Necronomicon': 71,
            'NeowsBlessing': 72,
            'Nilry\'s Codex': 73,
            'Nine Lives': 74,
            'Ninja Scroll': 75,
            'Nloth\'s Gift': 76,
            'Nuclear Battery': 77,
            'Nullstone Periapt': 78,
            'Odd Mushroom': 79,
            'Oddly Smooth Stone': 80,
            'Old Coin': 81,
            'Omamori': 82,
            'Orichalcum': 83,
            'Ornamental Fan': 84,
            'Orrery': 85,
            'Pandora\'s Box': 86,
            'Pantograph': 87,
            'Paper Crane': 88,
            'Paper Frog': 89,
            'Peace Pipe': 90,
            'Pear': 91,
            'Pen Nib': 92,
            'Philosopher\'s Stone': 93,
            'Potion Belt': 94,
            'Prayer Wheel': 95,
            'Question Card': 96,
            'Red Circlet': 97,
            'Red Mask': 98,
            'Red Skull': 99,
            'Regal Pillow': 100,
            'Ring of the Serpent': 101,
            'Ring of the Snake': 102,
            'Runic Capacitor': 103,
            'Runic Cube': 104,
            'Runic Dome': 105,
            'Runic Pyramid': 106,
            'Self Forming Clay': 107,
            'Shovel': 108,
            'Shuriken': 109,
            'Singing Bowl': 110,
            'Smiling Mask': 111,
            'Snake Skull': 112,
            'Snecko Eye': 113,
            'Sozu': 114,
            'Spirit Poop': 115,
            'Strange Spoon': 116,
            'Strawberry': 117,
            'Sundial': 118,
            'Symbiotic Virus': 119,
            'Test 1': 120,
            'Test 2': 121,
            'Test 3': 122,
            'Test 4': 123,
            'Test 5': 124,
            'Test 6': 125,
            'Test 7': 126,
            'Test 8': 127,
            'The Courier': 128,
            'The Specimen': 129,
            'Thread and Needle': 130,
            'Tingsha': 131,
            'Tiny Chest': 132,
            'Tiny House': 133,
            'Toolbox': 134,
            'Torii': 135,
            'Tough Bandages': 136,
            'Toxic Egg': 137,
            'Toxic Egg 2': 138,
            'Toy Ornithopter': 139,
            'Turnip': 140,
            'Unceasing Top': 141,
            'Vajra': 142,
            'Velvet Choker': 143,
            'War Paint': 144,
            'Whetstone': 145,
            'White Beast Statue': 146,
            'Boot': 147,
            'Mark of the Bloom': 148,
            'Busted Crown': 149,
            'Incense Burner': 150,
            'Empty Cage': 151,
            'Fusion Hammer': 152,
            'Coffee Dripper': 153,
            'SsserpentHead': 154,
            'MutagenicStrength': 155,
            'FaceOfCleric': 156,
            'NlothsMask': 157,
            'GremlinMask': 158,
            'CultistMask': 159,
            'TheAbacus': 160,
            'DollysMirror': 161,
            'ClockworkSouvenir': 162,
            'MealTicket': 163,
            'TwistedFunnel': 164,
            'HandDrill': 165,
            'HoveringKite': 166,
            'Sling': 167,
            'OrangePellets': 168,
            'WristBlade': 169,
            'StoneCalendar': 170,
            'Nunchaku': 171,
            'MawBank': 172,
            'Navi': 173
            }[relicId]

    @staticmethod
    def monsterIdToInt(monsterId):
        return{
            '': -1,
            'AcidSlime_L': 0,
            'AcidSlime_M': 1,
            'AcidSlime_S': 2,
            'Apology Slime': 3,
            'AwakenedOne': 4,
            'BanditBear': 5,
            'BanditChild': 6,
            'BanditLeader': 7,
            'BookOfStabbing': 8,
            'BronzeAutomaton': 9,
            'BronzeOrb': 10,
            'Byrd': 11,
            'Centurion': 12,
            'Champ': 13,
            'Chosen': 14,
            'CorruptHeart': 15,
            'Cultist': 16,
            'Darkling': 17,
            'Deca': 18,
            'Donu': 19,
            'Exploder': 20,
            'FireOrb': 21,
            'Dagger': 22,
            'FlameBruiser': 23,
            'Reptomancer': 24,
            'FungiBeast': 25,
            'FuzzyLouseDefensive': 26,
            'FuzzyLouseNormal': 27,
            'GiantHead': 28,
            'GremlinFat': 29,
            'GremlinLeader': 30,
            'GremlinNob': 31,
            'GremlinThief': 32,
            'GremlinTsundere': 33,
            'GremlinWarrior': 34,
            'GremlinWizard': 35,
            'Healer': 36,
            'Hexaghost': 37,
            'HexaghostBody': 38,
            'HexaghostOrb': 39,
            'JawWorm': 40,
            'Lagavulin': 41,
            'Looter': 42,
            'Maw': 43,
            'Mugger': 44,
            'Nemesis': 45,
            'Orb Walker': 46,
            'Puppeteer': 47,
            'Repulsor': 48,
            'Sentry': 49,
            'Serpent': 50,
            'Shelled Parasite': 51,
            'SlaverBlue': 52,
            'SlaverBoss': 53,
            'SlaverRed': 54,
            'SlimeBoss': 55,
            'SnakePlant': 56,
            'Snecko': 57,
            'SphericGuardian': 58,
            'SpikeSlime_L': 59,
            'SpikeSlime_M': 60,
            'SpikeSlime_S': 61,
            'Spiker': 62,
            'TheCollector': 63,
            'TheGuardian': 64,
            'TheGuardianOrb': 65,
            'TimeEater': 66,
            'TorchHead': 67,
            'Transient': 68
        }[monsterId]

    @staticmethod
    def orbIdToInt(orbId):
        return {
            '': -1,
            'Dark': 0,
            'Empty': 1,
            'Frost': 2,
            'Lightning': 3,
            'Plasma': 4
        }[orbId]

    @staticmethod
    def booleanToInt(boolean):
        return {
            '': -1,
            'False': 0,
            'True': 1
        }[boolean]

    @staticmethod
    def cardTypeToInt(cardType):
        return {
            '': -1,
            'NONE': -1,
            'ATTACK': 0,
            'SKILL': 1,
            'POWER': 2,
            'STATUS': 3,
            'CURSE': 4
        }[cardType]

    @staticmethod
    def intentToInt(intent):
        return {
            '': -1,
            'NONE': -1,
            'ATTACK': 0,
            'ATTACK_BUFF': 1,
            'ATTACK_DEBUFF': 2,
            'ATTACK_DEFEND': 3,
            'BUFF': 4,
            'DEBUFF': 5,
            'STRONG_DEBUFF': 6,
            'DEBUG': 7,
            'DEFEND': 8,
            'DEFEND_DEBUFF': 9,
            'DEFEND_BUFF': 10,
            'ESCAPE': 11,
            'MAGIC': 12,
            'SLEEP': 13,
            'STUN': 14,
            'UNKNOWN': 15
        }[intent]

    @staticmethod
    def roomTypeToInt(roomType):
        return {
            '': -1,
            'M': 0,
            'E': 1,
            'B': 2,
            '?': 3,
            '$': 4,
            'R': 5
        }[roomType]

    @staticmethod
    def potionIdToInt(potionId):
        return {
            '': -2,
            'Potion Slot': -1,
            'Ancient Potion': 0,
            'AttackPotion': 1,
            'Block Potion': 2,
            'BloodPotion': 3,
            'Dexterity Potion': 4,
            'Elixir': 5,
            'Energy Potion': 6,
            'EntropicBrew': 7,
            'EssenceOfSteel': 8,
            'Explosive Potion': 9,
            'FairyPotion': 10,
            'FearPotion': 11,
            'Fire Potion': 12,
            'FocusPotion': 13,
            'Fruit Juice': 14,
            'GamblersBrew': 15,
            'GhostInAJar': 16,
            'Health Potion': 17,
            'LiquidBronze': 18,
            'Poison Potion': 19,
            'PowerPotion': 20,
            'Regen Potion': 21,
            'SkillPotion': 22,
            'SmokeBomb': 23,
            'SneckoOil': 24,
            'SpeedPotion': 25,
            'SteroidPotion': 26,
            'Strength Potion': 27,
            'Swift Potion': 28,
            'Weak Potion': 29
        }[potionId]

    def fix_dict(self, tag, dict): #this acts weird, not sure why
        i=0
        for x in dict[tag]:
            corrected_dict = {(k + str(i)): v for k, v in dict.items()}
            dict.update(corrected_dict)
            i=i+1
        del dict[tag]
        return dict

    def createCombatDict(self, combatData):
        #data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\CombatDataDumpjsonDump.json"
        #with open(data_file, "r") as f:
        #    data = json.load(f)

        data = combatData

        i=0
        for enemy in data["jsonEnemyArrayList"]:
            #corrected_dict = {(k + str(i)): v for k, v in enemy.items()}
            data.update({(k + str(i)): v for k, v in enemy.items()})
            i=i+1
        del data["jsonEnemyArrayList"]

        i=0
        for power in data["enemypowers0"]:
            #corrected_dict = {("enemypowers0"+k + str(i)): v for k, v in power.items()}
            data.update({("enemypowers0"+k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["enemypowers0"]

        i=0
        for power in data["enemypowers1"]:
            #corrected_dict = {("enemypowers1"+k + str(i)): v for k, v in power.items()}
            data.update({("enemypowers1"+k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["enemypowers1"]

        i=0
        for power in data["enemypowers2"]:
            #corrected_dict = {("enemypowers2"+k + str(i)): v for k, v in power.items()}
            data.update({("enemypowers2"+k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["enemypowers2"]

        i=0
        for power in data["enemypowers3"]:
            #corrected_dict = {("enemypowers3"+k + str(i)): v for k, v in power.items()}
            data.update({("enemypowers3"+k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["enemypowers3"]

        i=0
        for power in data["enemypowers4"]:
            #corrected_dict = {("enemypowers4"+k + str(i)): v for k, v in power.items()}
            data.update({("enemypowers4"+k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["enemypowers4"]

        i=0
        for power in data["currentPowers"]:
            #corrected_dict = {(k + str(i)): v for k, v in power.items()}
            data.update({(k + str(i)): v for k, v in power.items()})
            i=i+1
        del data["currentPowers"]

        i=0
        for s in data["currentOrbs"]:
            #corrected_dict = {'currentOrbs'+str(i):s.split(',')}
            data.update({'currentOrbs'+str(i):s.split(',')})
            i=i+1
        del data["currentOrbs"]

        j=0
        for card in data["jsonCardArrayListHand"]:
            #corrected_dict = {("hand"+k + str(j)): v for k, v in card.items()}
            data.update({("hand"+k + str(j)): v for k, v in card.items()})
            j=j+1
        del data["jsonCardArrayListHand"]

        l=0
        for card in data["jsonCardArrayListExhaustPile"]:
            #corrected_dict = {("exhaust"+k + str(l)): v for k, v in card.items()}
            data.update({("exhaust"+k + str(l)): v for k, v in card.items()})
            l=l+1
        del data["jsonCardArrayListExhaustPile"]

        m=0
        for card in data["jsonCardArrayListDiscardPile"]:
            #corrected_dict = {("discard"+k + str(m)): v for k, v in card.items()}
            data.update({("discard"+k + str(m)): v for k, v in card.items()})
            m=m+1
        del data["jsonCardArrayListDiscardPile"]

        n=0
        for card in data["jsonCardArrayListDrawPile"]:
            #corrected_dict = {("draw"+k + str(n)): v for k, v in card.items()}
            data.update({("draw"+k + str(n)): v for k, v in card.items()})
            n=n+1
        del data["jsonCardArrayListDrawPile"]
        self.combatDataDict = data
        return data

    def combatDataToDF(self, combatData):
        data = self.createCombatDict(combatData)
        df = pds.DataFrame(dict([(k, pds.Series(v)) for k, v in data.items()]))
        df = df.fillna('')
        monster_id_cols = [col for col in df.columns if 'id' in col]
        df2 = df[monster_id_cols]
        for col in monster_id_cols:
            df[col] = df2[col].apply(self.monsterIdToInt)
        power_id_cols = [col for col in df.columns if 'powerId' in col]
        df2 = df[power_id_cols]
        for col in power_id_cols:
            df[col] = df2[col].apply(self.powerIdToInt)
        card_id_cols = [col for col in df.columns if 'cardId' in col]
        df2 = df[card_id_cols]
        for col in card_id_cols:
            df[col] = df2[col].apply(self.cardIdToInt)
        card_type_cols = [col for col in df.columns if 'type' in col]
        df2 = df[card_type_cols]
        for col in card_type_cols:
            df[col] = df2[col].apply(self.cardTypeToInt)
        intent_cols = [col for col in df.columns if ('intent' in col and len(col) == 7)]
        df2 = df[intent_cols]
        for col in intent_cols:
            df[col] = df2[col].apply(self.intentToInt)
        orb_id_cols = [col for col in df.columns if 'currentOrbs' in col]
        df2 = df[orb_id_cols]
        for col in orb_id_cols:
            df[col] = df2[col].apply(self.orbIdToInt)

        #for some reason, 9 extra rows get added. they only have blanks and -1's and all the data in row 0 look fine. so I will be dropping the last 9. expect a bug here.
        df = df[:1]
        return df

    def createStateDict(self, stateData):
        #data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\StateDataDumpjsonDump.json"
        #with open(data_file, "r") as f:
        #    data = json.load(f)

        data = stateData

        i=0
        for relic in data["relicsOwned"]:
            #corrected_dict = {("relicsOwned" + str(i)): v for v in relic.split(",")}
            data.update({("relicsOwned" + str(i)): v for v in relic.split(",")})
            i=i+1
        del data["relicsOwned"]

        i=0
        for relic in data["relicsSeen"]:
            #corrected_dict = {("relicsSeen" + str(i)): v for v in relic.split(",")}
            data.update({("relicsSeen" + str(i)): v for v in relic.split(",")})
            i=i+1
        del data["relicsSeen"]

        i=0
        for card in data["deck"]:
            #corrected_dict = {("deck" + str(i)): v for v in card.split(",")}
            data.update({("deck" + str(i)): v for v in card.split(",")})
            i=i+1
        del data["deck"]

        j=0
        for potion in data["potions"]:
            #corrected_dict = {("potions" + str(j)): v for v in potion.split(",")}
            data.update({("potions" + str(j)): v for v in potion.split(",")})
            j=j+1
        del data["potions"]

        self.stateDataDict = data
        return data

    def stateDataToDF(self, stateData):
        data = self.createStateDict(stateData)
        df = pds.DataFrame(dict([(k, pds.Series(v)) for k, v in data.items()]))
        df = df.fillna('')
        relic_id_cols = [col for col in df.columns if 'relic' in col]
        df2 = df[relic_id_cols]
        for col in relic_id_cols:
            df[col] = df2[col].apply(self.relicIdToInt)
        card_id_cols = [col for col in df.columns if 'deck' in col]
        df2 = df[card_id_cols]
        for col in card_id_cols:
            df[col] = df2[col].apply(self.cardIdToInt)
        potion_id_cols = [col for col in df.columns if 'potions' in col]
        df2 = df[potion_id_cols]
        for col in potion_id_cols:
            df[col] = df2[col].apply(self.potionIdToInt)
        df['roomType'] = df['roomType'].apply(self.roomTypeToInt)
        df['chosenClass'] = df['chosenClass'].apply(self.classNameToInt)


        #for some reason, 9 extra rows get added. they only have blanks and -1's and all the data in row 0 look fine. so I will be dropping the last 9. expect a bug here.
        df = df[:1]
        return df

    def get_reward(self):
        if self.previous_state.empty:
            reward = 0
        else:
            health_reward = self.current_state['currentHealth'] - self.previous_state['currentHealth']
            enemy_health_reward = self.previous_state['currentHealth0'] - self.current_state['currentHealth0'] + self.previous_state['currentHealth1'] - self.current_state['currentHealth1'] + self.previous_state['currentHealth2'] - self.current_state['currentHealth2'] + self.previous_state['currentHealth3'] - self.current_state['currentHealth3'] + self.previous_state['currentHealth4'] - self.current_state['currentHealth4']
            # currentGoldReward = self.current_state['currentGold'] / 10
            # floorNumReward = self.current_state['floorNum']
            # actNumReward = state['actNum'] * 5 #maybe this should multiply all other rewards instead of being its own constant reward?
            # relic_id_cols = [col for col in state.columns if 'relic' in col]
            # relicReward = state.groupby(relic_id_cols).ngroups
            reward = health_reward + enemy_health_reward # + currentGoldReward + floorNumReward + actNumReward + relicReward
        print("current reward: ", int(reward))
        self.remember(self.previous_state, self.previous_action, reward, self.current_state, False)
        return reward

    def discount_rewards(self, r, gamma=0.99):
        """Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        prior = 0
        out = []
        for val in r:
            new_val = val + prior * gamma
            out.append(new_val)
            prior = new_val
        return np.array(out[::-1])

    def custom_loss(self, y_true, y_pred):
        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        return K.mean(log_lik, keepdims=True)

    # Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error*error / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

    def copy_model(self, model):
        """Returns a copy of a keras model."""
        import keras.models
        model.save('tmp_model')
        return keras.models.load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})

    def create_combined_dataframe(self, stateData, combatData):
        stateDf = self.stateDataToDF(stateData)
        combatDf = self.combatDataToDF(combatData)
        result = pds.concat([stateDf, combatDf], axis=1, join='inner')
        result = result.drop('gameID', 1)
        result = result.drop('combatStateID', 1)
        result = result.drop('currentStateID', 1)
        self.previous_state_array = self.current_state_array
        self.previous_state = self.current_state
        self.current_state = result
        self.current_state_array = self.current_state.values
        return result, combatDf

    def create_combat_action_space(self, state):
        hand_cols = [col for col in state.columns if 'handcardId' in col]
        cards_in_hand = state[hand_cols]
        monster_id_cols = [col for col in state.columns if 'id' in col]
        monsters_to_target = state[monster_id_cols]
        potion_id_cols = [col for col in state.columns if 'potions' in col]
        potions_available = state[potion_id_cols]
        action_cols = hand_cols+potion_id_cols
        combination = [(x+y)for x in action_cols for y in monster_id_cols]
        actions = [x for x in action_cols]
        monsters = [y for y in monster_id_cols]
        combination.append('end_turn')
        actions.append('end_turn')
        df = pds.DataFrame([range(len(combination))])
        df_actions = pds.DataFrame([range(len(actions))])
        df_monsters = pds.DataFrame([range(len(monsters))])
        df.columns = combination
        return df, df_actions, df_monsters

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def create_model(self):

        from keras.models import Model
        from keras.layers import Dense, Input
        from keras.optimizers import RMSprop
        S = Input(shape=[5951, ])
        h0 = Dense(1000, activation="relu")(S)
        h1 = Dense(500, activation="relu")(h0)
        Target = Dense(5, activation="softmax")(h1)
        Action = Dense(16, activation="softmax")(h1)
        model = Model(inputs=S, outputs=[Target, Action])
        model.summary()
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=self.huber_loss)
        global graph
        graph = tf.get_default_graph()
        return model

    def predict_combat_action(self, data):
        #TODO: call model.fit(state,action) with numpy arrays of the state and the action somewhere?
        global graph
        with graph.as_default():
            act_values = self.model.predict(data)
        if self.turn != 1:
            current_reward = self.get_reward()
            #self.model.fit(self.current_state, self.action_list)
        self.turn = self.turn+1
        m = self.get_valid_monster(act_values)
        c = self.get_valid_action(act_values)
        monsters = np.zeros(5)
        monsters[m] = 1
        cards = np.zeros(16)
        cards[c] = 1
        self.action_list = []
        self.action_list.append(monsters)
        self.action_list.append(cards)
        self.previous_action = self.current_action
        self.current_action = np.append(m, c)
        return m, c

    def get_valid_action(self, predicted_action_values):
        cards = {k: v for k, v in self.combatDataDict.items() if 'handisPlayable' in k}
        playable_cards = np.fromiter(cards.values(), dtype=int)

        potions = {k: 1 if v != "" and v != "Potion Slot" else 0 for k, v in self.stateDataDict.items() if 'potions' in k}
        playable_potions = np.fromiter(potions.values(), dtype=int)

        playable_cards_and_actions = np.append(playable_cards, playable_potions)
        playable_cards_and_actions = np.append(playable_cards_and_actions, np.array([1]))

        final_action_values = np.multiply(playable_cards_and_actions, predicted_action_values[1])

        return np.argmax(final_action_values[0])

    def get_valid_monster(self, predicted_action_values):
        monsters = {k: 1 if v != 0 else 0 for k, v in self.combatDataDict.items() if 'currentHealth' in k}
        targetable_monsters = np.fromiter(monsters.values(), dtype=int)

        final_action_values = np.multiply(targetable_monsters, predicted_action_values[0])

        return np.argmax(final_action_values[0])

    def get_rand_valid_action(self):
        cards = {k: v for k, v in self.combatDataDict.items() if 'handisPlayable' in k}
        playable_cards = np.fromiter(cards.values(), dtype=int)

        potions = {k: 1 if v != "" and v != "Potion Slot" else 0 for k, v in self.stateDataDict.items() if 'potions' in k}
        playable_potions = np.fromiter(potions.values(), dtype=int)

        playable_cards_and_actions = np.append(playable_cards, playable_potions)
        playable_cards_and_actions = np.append(playable_cards_and_actions, np.array([1]))

        final_action_value_index = np.random.choice(np.nonzero(playable_cards_and_actions == 1)[0], replace=False)
        final_action_values = np.zeros(playable_cards_and_actions.shape)
        final_action_values[final_action_value_index] = 1

        return final_action_values

    def get_rand_valid_monster(self):
        monsters = {k: 1 if v != 0 else 0 for k, v in self.combatDataDict.items() if 'currentHealth' in k}
        targetable_monsters = np.fromiter(monsters.values(), dtype=int)

        if np.nonzero(targetable_monsters == 1)[0].size == 0:
            final_action_values = np.zeros(targetable_monsters.shape)
            final_action_values[0] = 1
        else:
            final_action_value_index = np.random.choice(np.nonzero(targetable_monsters == 1)[0], replace=False)
            final_action_values = np.zeros(targetable_monsters.shape)
            final_action_values[final_action_value_index] = 1

        return final_action_values
