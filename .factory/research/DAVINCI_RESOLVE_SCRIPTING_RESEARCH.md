# DaVinci Resolve Scripting Research Report

## 1. Python Scripting API Basics

### Connecting to DaVinci Resolve

**DaVinci Resolve MUST be running** for the Python scripting API to work. The API connects to a live Resolve instance.

#### Environment Variables (macOS)
```bash
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"
```

#### Basic Connection Script
```python
#!/usr/bin/env python
import DaVinciResolveScript as dvr_script

resolve = dvr_script.scriptapp("Resolve")
projectManager = resolve.GetProjectManager()
project = projectManager.GetCurrentProject()
```

#### Headless Mode
Resolve can run without UI: launch with `-nogui` flag. The scripting API still works fully.

### Free vs Studio Version
- **DaVinci Resolve Studio**: Full scripting API support, including external scripts run from command line
- **DaVinci Resolve Free**: Scripting works **only from within Resolve** (Scripts menu or Console). External command-line scripting does NOT work. Workaround: use `resolve = app.GetResolve()` instead of `DaVinciResolveScript.scriptapp("Resolve")` when running from Scripts menu in Free version
- Scripting API has been available since **DaVinci Resolve 15+** (Python & Lua). Current docs are for **v20.3**
- Supported Python versions: Python 3.6+ (64-bit). Recent versions support Python 3.10+

---

## 2. Key API Methods for Timeline Creation and Clip Placement

### Object Hierarchy
```
Resolve
├── GetProjectManager() → ProjectManager
│   ├── CreateProject(name) → Project
│   └── GetCurrentProject() → Project
│       ├── GetMediaPool() → MediaPool
│       │   ├── GetRootFolder() → Folder
│       │   ├── ImportMedia([paths]) → [MediaPoolItem]
│       │   ├── CreateEmptyTimeline(name) → Timeline
│       │   ├── CreateTimelineFromClips(name, [clips]) → Timeline
│       │   ├── CreateTimelineFromClips(name, [{clipInfo}]) → Timeline
│       │   ├── AppendToTimeline([clips]) → [TimelineItem]
│       │   ├── AppendToTimeline([{clipInfo}]) → [TimelineItem]
│       │   └── ImportTimelineFromFile(filePath, {importOptions}) → Timeline
│       └── GetCurrentTimeline() → Timeline
└── GetMediaStorage() → MediaStorage
    └── AddItemListToMediaPool([paths]) → [MediaPoolItem]
```

### Complete Workflow Example
```python
import DaVinciResolveScript as dvr_script

# 1. Connect
resolve = dvr_script.scriptapp("Resolve")
projectManager = resolve.GetProjectManager()

# 2. Create project
project = projectManager.CreateProject("MyProject")

# 3. Get media pool
mediaPool = project.GetMediaPool()

# 4. Import media files
clips = mediaPool.ImportMedia([
    "/path/to/clip1.mp4",
    "/path/to/clip2.mp4",
    "/path/to/clip3.mp4"
])

# 5a. Create timeline from clips (simple - full clips in order)
timeline = mediaPool.CreateTimelineFromClips("My Timeline", clips)

# 5b. OR create timeline with specific in/out points
clipInfos = [
    {"mediaPoolItem": clips[0], "startFrame": 0, "endFrame": 150},
    {"mediaPoolItem": clips[1], "startFrame": 30, "endFrame": 200},
    {"mediaPoolItem": clips[2], "startFrame": 10, "endFrame": 100},
]
timeline = mediaPool.CreateTimelineFromClips("My Timeline", clipInfos)

# 6. Append more clips to existing timeline
project.SetCurrentTimeline(timeline)
moreClipInfos = [
    {
        "mediaPoolItem": clips[0],
        "startFrame": 200,
        "endFrame": 350,
        "mediaType": 1  # 1=Video only, 2=Audio only
    }
]
mediaPool.AppendToTimeline(moreClipInfos)

# 7. Save
projectManager.SaveProject()
```

### Importing a Timeline from EDL/XML/AAF
```python
# Import timeline from file (EDL, FCPXML, AAF, etc.)
importOptions = {
    "timelineName": "Imported Timeline",
    "importSourceClips": True,
    "sourceClipsPath": "/path/to/media/folder"
}
timeline = mediaPool.ImportTimelineFromFile("/path/to/timeline.edl", importOptions)
```

### Exporting a Timeline
```python
timeline = project.GetCurrentTimeline()

# Export as EDL
timeline.Export("/path/to/output.edl", resolve.EXPORT_EDL, resolve.EXPORT_NONE)

# Export as FCPXML 1.8
timeline.Export("/path/to/output.fcpxml", resolve.EXPORT_FCPXML_1_8, resolve.EXPORT_NONE)

# Export as FCP 7 XML (compatible with Premiere Pro)
timeline.Export("/path/to/output.xml", resolve.EXPORT_FCP_7_XML, resolve.EXPORT_NONE)
```

### Supported Export Types
- `resolve.EXPORT_EDL` (requires subtypes: EXPORT_CDL, EXPORT_SDL, EXPORT_MISSING_CLIPS, EXPORT_NONE)
- `resolve.EXPORT_FCP_7_XML`
- `resolve.EXPORT_FCPXML_1_3` through `resolve.EXPORT_FCPXML_1_10`
- `resolve.EXPORT_AAF` (requires subtypes: EXPORT_AAF_NEW, EXPORT_AAF_EXISTING)
- `resolve.EXPORT_TEXT_CSV`, `resolve.EXPORT_TEXT_TAB`

---

## 3. EDL (Edit Decision List) Format — CMX 3600

### Format Specification

EDL is a plain-text format. Each line represents an edit event. Maximum 999 events in CMX 3600.

#### Structure
```
TITLE: Timeline Name
FCM: NON-DROP FRAME

[event#] [reel] [track] [transition] [src_in] [src_out] [rec_in] [rec_out]
```

#### Column Definitions
| Column | Description |
|--------|-------------|
| Event # | 001-999, sequential edit number |
| Reel/Source | Source tape/file name (max 8 chars in CMX3600) |
| Track | V=Video, A=Audio, A2=Audio Ch2, etc. |
| Transition | C=Cut, D nnn=Dissolve (nnn frames), W### nnn=Wipe |
| Source In | Timecode of source clip start |
| Source Out | Timecode of source clip end (1 frame AFTER last shown) |
| Record In | Timecode on master/output timeline |
| Record Out | Timecode on master/output timeline end |

#### FCM (Frame Code Mode)
- `NON-DROP FRAME` — for 24fps, 25fps, 30fps NDF
- `DROP FRAME` — for 29.97fps DF

### Concrete EDL Example
```
TITLE: My Cut List
FCM: NON-DROP FRAME

001  clip1    V     C        00:00:05:00 00:00:10:00 01:00:00:00 01:00:05:00
* FROM CLIP NAME: interview_wide.mp4
002  clip2    V     C        00:00:02:12 00:00:08:00 01:00:05:00 01:00:10:12
* FROM CLIP NAME: broll_sunset.mp4
003  clip3    V     C        00:00:00:00 00:00:03:15 01:00:10:12 01:00:14:02
* FROM CLIP NAME: interview_closeup.mp4
004  clip1    V     C        00:00:15:00 00:00:22:10 01:00:14:02 01:00:21:12
* FROM CLIP NAME: interview_wide.mp4
005  clip2    V     D  025   00:00:12:00 00:00:18:00 01:00:21:12 01:00:27:12
* FROM CLIP NAME: broll_sunset.mp4
```

**Key notes:**
- Lines starting with `*` are comments (used for long clip names, notes)
- Out points are exclusive (1 frame after last visible frame)
- Record In of each event = Record Out of previous event (for cuts)
- Dissolves/wipes use 2 lines (FROM clip + TO clip)
- Audio tracks: `V` = video only; `A` = audio ch1; `AA/V` or `A1A2V` = video + 2 audio channels
- Speed changes use `M2` modifier line after the event

### Generating an EDL Programmatically (Python)
```python
def generate_edl(title, clips, fps=24):
    """
    clips: list of dicts with keys:
        - name: str (max 8 chars for reel)
        - src_in: str timecode "HH:MM:SS:FF"
        - src_out: str timecode
        - full_name: str (for comment)
    """
    lines = []
    lines.append(f"TITLE: {title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")
    
    rec_tc = "01:00:00:00"  # starting record timecode
    
    for i, clip in enumerate(clips, 1):
        reel = clip["name"][:8].ljust(8)
        event = str(i).zfill(3)
        
        # Calculate record out from source duration
        rec_out = add_timecodes(rec_tc, subtract_timecodes(clip["src_out"], clip["src_in"], fps), fps)
        
        lines.append(f"{event}  {reel} V     C        {clip['src_in']} {clip['src_out']} {rec_tc} {rec_out}")
        if clip.get("full_name"):
            lines.append(f"* FROM CLIP NAME: {clip['full_name']}")
        
        rec_tc = rec_out
    
    return "\n".join(lines)
```

---

## 4. FCPXML Format (for DaVinci Resolve Import)

DaVinci Resolve supports import of FCPXML versions 1.3 through 1.10, and also FCP 7 XML (xmeml).

### FCPXML Structure Overview

FCPXML uses rational time (fractions of seconds) for all durations and positions:
- `1/24s` = one frame at 24fps
- `100/2400s` = same thing
- `1001/30000s` = one frame at 29.97fps

### Concrete FCPXML 1.8 Example

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
        <format id="r1" name="FFVideoFormat1080p24" 
                frameDuration="100/2400s" width="1920" height="1080"/>
        <asset id="r2" name="interview_wide" uid="A1B2C3D4" 
               src="file:///Users/me/Footage/interview_wide.mp4"
               start="0/1s" duration="1800/24s" 
               hasVideo="1" format="r1" hasAudio="1" 
               audioSources="1" audioChannels="2" audioRate="48000"/>
        <asset id="r3" name="broll_sunset" uid="E5F6G7H8" 
               src="file:///Users/me/Footage/broll_sunset.mp4"
               start="0/1s" duration="720/24s"
               hasVideo="1" format="r1" hasAudio="1"
               audioSources="1" audioChannels="2" audioRate="48000"/>
        <asset id="r4" name="interview_closeup" uid="I9J0K1L2"
               src="file:///Users/me/Footage/interview_closeup.mp4"
               start="0/1s" duration="600/24s"
               hasVideo="1" format="r1" hasAudio="1"
               audioSources="1" audioChannels="2" audioRate="48000"/>
    </resources>
    <library>
        <event name="My Event">
            <project name="My Cut">
                <sequence format="r1" duration="600/24s"
                          tcStart="0/1s" tcFormat="NDF"
                          audioLayout="stereo" audioRate="48k">
                    <spine>
                        <!-- Clip 1: interview_wide, frames 120-240 (5s-10s) -->
                        <asset-clip ref="r2" name="interview_wide"
                                    offset="0/1s"
                                    start="120/24s" duration="120/24s"
                                    format="r1" audioRole="dialogue"/>
                        
                        <!-- Clip 2: broll_sunset, frames 60-192 -->
                        <asset-clip ref="r3" name="broll_sunset"
                                    offset="120/24s"
                                    start="60/24s" duration="132/24s"
                                    format="r1" audioRole="dialogue"/>
                        
                        <!-- Clip 3: interview_closeup, frames 0-87 -->
                        <asset-clip ref="r4" name="interview_closeup"
                                    offset="252/24s"
                                    start="0/24s" duration="87/24s"
                                    format="r1" audioRole="dialogue"/>
                        
                        <!-- Clip 4: interview_wide, frames 360-538 -->
                        <asset-clip ref="r2" name="interview_wide"
                                    offset="339/24s"
                                    start="360/24s" duration="178/24s"
                                    format="r1" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
```

### Key FCPXML Elements

| Element | Purpose |
|---------|---------|
| `<resources>` | Container for format definitions and asset (media file) references |
| `<format>` | Defines video format (resolution, frame rate via `frameDuration`) |
| `<asset>` | Defines a media file (path via `src`, duration, audio/video properties) |
| `<library>` | Top-level container |
| `<event>` | Organizational container (like a bin) |
| `<project>` | Contains a sequence/timeline |
| `<sequence>` | The timeline itself |
| `<spine>` | The primary storyline (main track) |
| `<asset-clip>` | A clip on the timeline referencing an asset |

### Key Attributes for `<asset-clip>`

| Attribute | Meaning |
|-----------|---------|
| `ref` | References an asset `id` in `<resources>` |
| `offset` | Position on the timeline (from timeline start) |
| `start` | In-point within the source media |
| `duration` | Duration of the clip on timeline |
| `name` | Display name |

### FCP 7 XML (xmeml) — Alternative Format

DaVinci Resolve also imports FCP 7 XML (`resolve.EXPORT_FCP_7_XML`). This is the same format Premiere Pro uses. It uses track-based structure with `<sequence>` → `<media>` → `<video>` → `<track>` → `<clipitem>` hierarchy. Frame counts are integers (not rational seconds), which can be simpler to generate.

---

## 5. Gotchas and Limitations

1. **Resolve MUST be running**: The Python API connects to a live instance. No offline/standalone processing possible.

2. **Free version scripting is limited**: External command-line scripts don't work in Free. Only scripts run from Resolve's Scripts menu or Console work, using `app.GetResolve()` workaround.

3. **Reel names in EDL**: CMX3600 limits reel names to 8 characters. Longer names need `* FROM CLIP NAME:` comments.

4. **EDL limitations**: Max 999 events, max 4 audio channels, no nested timelines, limited metadata.

5. **FCPXML version compatibility**: DaVinci Resolve's supported FCPXML versions may lag behind Final Cut Pro's latest. Resolve supports up to 1.10 (as of v20.3). Always check your Resolve version's supported formats.

6. **Rational time in FCPXML**: All times are rational fractions of seconds (e.g., `100/2400s`), not frame numbers. This is mathematically precise but requires careful conversion.

7. **Frame numbering**: 
   - EDL out-points are **exclusive** (1 frame after the last visible frame)
   - FCPXML uses start + duration model
   - API `startFrame`/`endFrame` in `AppendToTimeline` clipInfo dicts

8. **Media paths**: FCPXML uses `file:///` URI format with URL-encoded spaces (`%20`). EDL uses reel names that must be mapped to actual files.

9. **`AppendToTimeline` is append-only**: You can only add clips to the end of a timeline. To insert at arbitrary positions, use `ImportTimelineFromFile` with EDL/XML, or manually manipulate via the API after appending.

10. **Project settings**: Timeline frame rate, resolution, etc. must match or be set before importing/creating timelines. Use `project.SetSetting()`.

11. **No direct "insert at position" API**: The scripting API doesn't provide a method to insert a clip at a specific timeline position. You must either:
    - Build the timeline in order using `AppendToTimeline` with `startFrame`/`endFrame`
    - Generate an EDL or FCPXML file and use `ImportTimelineFromFile`

---

## 6. Recommended Approach for Automated Timeline Creation

### Option A: Generate EDL file → Import into Resolve
- **Pros**: Simple text format, easy to generate, universally supported
- **Cons**: Limited to 999 events, 8-char reel names, no metadata beyond comments
- **Best for**: Simple cut lists with sequential clips

### Option B: Generate FCPXML → Import into Resolve  
- **Pros**: Rich metadata, no event limit, precise timing, supports transitions
- **Cons**: More complex XML structure, rational time math
- **Best for**: Complex timelines with metadata, multiple tracks

### Option C: Direct API scripting
- **Pros**: Most control, can set clip properties, markers, etc.
- **Cons**: Requires Resolve running, append-only timeline building, Studio for external scripts
- **Best for**: Automated workflows where Resolve is part of a pipeline

### For a cut list tool, **Option A (EDL) or Option B (FCPXML)** is recommended:
- Generate the file offline (no Resolve needed)
- Import into Resolve manually or via API
- User can also import into Premiere Pro, Final Cut Pro, or any other NLE
