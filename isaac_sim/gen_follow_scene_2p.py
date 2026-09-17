#!/usr/bin/env python3
"""Build the two-person room: one in purple, one in yellow.

The single-person scene (gen_follow_scene.py) cannot support a
language-conditioned task. With one person in the room the instruction has
nothing to select between, so a VLA trained there learns to ignore its own text
input -- measured: swapping "follow the person" for "stop immediately and do not
move", or for the nonsense string "banana telescope kettle", moved the commanded
velocity by 0.03-0.06 against a 0.34 spread across frames, and the nonsense
string perturbed it as much as the contradictory order did. That is not weak
language grounding, it is no grounding at all, and it is a property of the
DATASET, not of the architecture.

This scene fixes the cause: two people who differ in exactly one visible
attribute, so "follow the person in purple" and "follow the person in yellow"
have different correct answers from the same camera image.

Colouring: the character's clothing is split across separate materials (shirt,
vest, jeans, shoes, skin), each an MDL shader carrying a `diffuse_tint` input.
The asset ships no texture files -- the referenced PNGs 404 -- so the tint is
not modulating a pattern, it lands as a flat colour, which is what this task
wants. Shirt and vest are tinted (the torso is what the camera sees at
following range); jeans, shoes and skin are left alone so the two figures stay
recognisably people rather than coloured statues.

The references must NOT be instanceable. Instanced prims share one prototype,
so a per-instance material override silently does nothing to either copy -- both
people would come out the same colour, with no error to say why.

    /home/itri/anaconda3/envs/env_isaaclab/bin/python gen_follow_scene_2p.py \
        --out /home/itri/mir_isaac_test/scenes/follow_room_2p.usd
"""
import argparse
import json
import os

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

# Clothing materials to tint, matched by name anywhere under the referenced
# body. Trousers and shoes are included as well as the top, because the colour
# is the ONLY thing that tells the two people apart and there was not enough of
# it: measured on the first build, which tinted the shirt and vest alone, a
# person carried 200-500 saturated pixels in a 307200-pixel frame -- 0.1 % --
# and the platform camera's median was 0, i.e. it saw neither person at all in
# most frames. Tinting the whole outfit roughly doubles the coloured area. Searched rather than addressed by path: referencing flattens the source
# file's defaultPrim away, so the depth the materials end up at is a property of
# how the asset was authored, not something to hard-code. The guard below turns
# a failed match into a stop, so a wrong guess cannot quietly produce two people
# in the same colour.
TINT_MATERIALS = ("opaque__fabric__shirt", "opaque__fabric__vest",
                  "opaque__fabric__bluejeans", "opaque__leather__tennisshoes")

# Saturated and far apart in hue, so the two are separable at 5 m in a 640x480
# frame, not just under close inspection.
COLOURS = {
    "purple": (0.42, 0.11, 0.72),
    "yellow": (0.95, 0.78, 0.06),
    # Never in any training data. For testing whether a colour word the policy
    # was not trained on still selects the right person.
    "red": (0.80, 0.08, 0.06),
}


def material(stage, path, rgb, roughness=0.55):
    """UsdPreviewSurface, not displayColor: the latter does not show under RTX."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def box(stage, path, center, size, mat=None, collision=True):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    c.GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    x = UsdGeom.XformCommonAPI(c)
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*size))
    if mat is not None:
        UsdShade.MaterialBindingAPI(c.GetPrim()).Apply(c.GetPrim())
        UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def add_person(stage, name, usd_path, start, rgb):
    """Reference the character under a clean parent, then tint its clothes.

    Parent/child split as in the single-person scene: the character USD declares
    xformOp:rotateXYZ as double while XformCommonAPI.SetRotate only takes
    GfVec3f, so the prim that is referenced onto cannot also be the prim that is
    driven. The parent is what the walker moves at run time.
    """
    root = f"/World/{name}"
    person = UsdGeom.Xform.Define(stage, root)
    body = UsdGeom.Xform.Define(stage, root + "/Body")
    body.GetPrim().GetReferences().AddReference(os.path.abspath(usd_path))
    body.GetPrim().SetInstanceable(False)      # see module docstring
    UsdGeom.XformCommonAPI(person).SetTranslate(Gf.Vec3d(start[0], start[1], 0.0))
    UsdGeom.XformCommonAPI(person).SetRotate(
        Gf.Vec3f(0.0, 0.0, float(start[2])), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    tinted = []
    if rgb is None:
        return root, tinted
    for prim in Usd.PrimRange(body.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        # the shader sits under the material, so the material is its parent
        mat_name = prim.GetParent().GetName()
        if mat_name not in TINT_MATERIALS:
            continue
        sh = UsdShade.Shader(prim)
        inp = sh.GetInput("diffuse_tint")
        if not inp:
            inp = sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
        inp.Set(Gf.Vec3f(*rgb))
        tinted.append(mat_name)
    return root, tinted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--room", type=float, nargs=4, default=[-8.0, 8.0, -6.0, 6.0],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--person",
                    default="/home/itri/mir_isaac_test/people/F_Business_02.usd")
    ap.add_argument("--purple-start", type=float, nargs=3, default=[3.0, 2.0, 0.0],
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--yellow-start", type=float, nargs=3, default=[3.0, -2.0, 0.0],
                    metavar=("X", "Y", "YAW"),
                    help="Both start ahead of the robot and to opposite sides, "
                         "so the very first frame of every episode already "
                         "poses the question the instruction has to answer.")
    ap.add_argument("--no-tint", action="store_true",
                    help="leave both characters in their original clothing. "
                         "For testing a policy trained on ONE person against a "
                         "scene containing two: the follow models were trained "
                         "on the untinted character, so a purple and a yellow "
                         "one change the appearance as well as the count, and a "
                         "failure could not be attributed to either.")
    ap.add_argument("--pillars", type=int, default=0,
                    help="0 by default. With two people to keep apart, "
                         "occlusion by pillars adds a second failure mode on "
                         "top of the one being measured.")
    ap.add_argument("--swap-colours", action="store_true",
                    help="dress /World/PersonPurple in the yellow tint and "
                         "/World/PersonYellow in the purple one, everything else "
                         "unchanged. For testing a trained policy: the prim names, "
                         "TF frames and walker indices stay put while the colour "
                         "moves, so a policy that follows the named COLOUR now "
                         "tracks the other prim, and one that keyed on anything "
                         "else keeps tracking the same prim.")
    ap.add_argument("--wear", action="append", default=[], metavar="PRIM=COLOUR",
                    help="dress one prim in any colour from COLOURS, e.g. "
                         "--wear yellow=red puts the red tint on /World/PersonYellow. "
                         "Prim names and TF frames stay purple/yellow whatever they "
                         "wear; the layout json records it under \"wears\". "
                         "Overrides --swap-colours for that prim.")
    args = ap.parse_args()

    if not os.path.exists(args.person):
        raise SystemExit(f"character USD not found: {args.person}")

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    looks = {
        "floor": material(stage, "/World/Looks/floor", (0.35, 0.35, 0.34), 0.9),
        "wall": material(stage, "/World/Looks/wall", (0.60, 0.59, 0.56), 0.9),
        "pillar": material(stage, "/World/Looks/pillar", (0.45, 0.44, 0.42), 0.8),
    }

    xmin, xmax, ymin, ymax = args.room
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, d, t, hgt = xmax - xmin, ymax - ymin, 0.15, 2.8
    box(stage, "/World/Floor", (cx, cy, -0.05), (w + 2 * t, d + 2 * t, 0.10),
        looks["floor"])
    for nm, ctr, sz in (
            ("north", (cx, ymax + t / 2, hgt / 2), (w + 2 * t, t, hgt)),
            ("south", (cx, ymin - t / 2, hgt / 2), (w + 2 * t, t, hgt)),
            ("west", (xmin - t / 2, cy, hgt / 2), (t, d, hgt)),
            ("east", (xmax + t / 2, cy, hgt / 2), (t, d, hgt))):
        box(stage, f"/World/Walls/{nm}", ctr, sz, looks["wall"])

    pillars = []
    spots = [(-3.5, 3.0), (3.5, 3.0), (-3.5, -3.0), (3.5, -3.0)]
    for i, (px, py) in enumerate(spots[:args.pillars]):
        box(stage, f"/World/Pillars/p{i}", (px, py, hgt / 2), (0.45, 0.45, hgt),
            looks["pillar"])
        pillars.append([px, py, 0.45])

    people = {}
    for name, start in (("PersonPurple", args.purple_start),
                        ("PersonYellow", args.yellow_start)):
        colour = "purple" if "Purple" in name else "yellow"
        # The key stays the prim's name; "wears" is the tint actually applied,
        # so a swapped layout says so instead of silently lying.
        wears = ({"purple": "yellow", "yellow": "purple"}[colour]
                 if args.swap_colours else colour)
        wear_map = dict(w.split("=", 1) for w in args.wear)
        if colour in wear_map:
            wears = wear_map[colour]
            if wears not in COLOURS:
                raise SystemExit(f"--wear {colour}={wears}: unknown colour, "
                                 f"have {sorted(COLOURS)}")
        root, tinted = add_person(stage, name, args.person, start,
                                  None if args.no_tint else COLOURS[wears])
        people[colour] = {"prim": root, "start": list(start),
                          "rgb": list(COLOURS[wears]), "tinted": tinted}
        if wears != colour:
            people[colour]["wears"] = wears
        print(f"  {colour:<7} {root}  wears {wears}  tinted {tinted}")
        if not tinted and not args.no_tint:
            # Loud: a person that failed to tint is the same colour as the other
            # one, and the whole experiment silently becomes the single-person
            # one again.
            raise SystemExit(f"{name}: no clothing material was tinted -- the "
                             f"character layout under {CHAR_ROOT} has changed")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(200.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/key")
    key.CreateIntensityAttr(700.0)
    key.CreateAngleAttr(1.5)
    UsdGeom.XformCommonAPI(key).SetRotate((-45.0, 0.0, 25.0),
                                          UsdGeom.XformCommonAPI.RotationOrderXYZ)

    stage.GetRootLayer().Save()

    layout = {"room": args.room, "people": people, "pillars": pillars,
              "person_usd": os.path.abspath(args.person), "wall_height": hgt}
    with open(os.path.splitext(args.out)[0] + "_layout.json", "w") as f:
        json.dump(layout, f, indent=2)
    print(f"wrote {args.out}")
    _app.close()


if __name__ == "__main__":
    main()
