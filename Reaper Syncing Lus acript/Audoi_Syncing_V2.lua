-- REAPER Script: Sync Segments to New Timestamps
-- Instructions: Select the source item, run script, and choose your text file.

function msg(m) reaper.ShowConsoleMsg(tostring(m) .. "\n") end

function hms_to_seconds(hms)
    local h, m, s, ms = hms:match("(%d+):(%d+):(%d+),(%d+)")
    if not h then return nil end
    return (tonumber(h) * 3600) + (tonumber(m) * 60) + tonumber(s) + (tonumber(ms) / 1000)
end

function ms_to_seconds(ms)
    if not ms then return nil end
    return tonumber(ms) / 1000
end

function main()
    local sel_item = reaper.GetSelectedMediaItem(0, 0)
    if not sel_item then
        reaper.ShowMessageBox("Please select the original media item first.", "Error", 0)
        return
    end

    local retval, file_path = reaper.GetUserFileNameForRead("", "Select Timestamp File", ".txt")
    if not retval then return end

    local file = io.open(file_path, "r")
    if not file then return end

    -- Setup Tracks
    local src_track = reaper.GetMediaItem_Track(sel_item)
    local src_track_idx = reaper.GetMediaTrackInfo_Value(src_track, "IP_TRACKNUMBER")
    
    -- Insert new track below
    reaper.InsertTrackAtIndex(src_track_idx, true)
    local dest_track = reaper.GetTrack(0, src_track_idx)
    reaper.GetSetMediaTrackInfo_String(dest_track, "P_NAME", "Synced Audio", true)

    reaper.Undo_BeginBlock()

    for line in file:lines() do
        local start_orig, end_orig, start_sync, duration

        -- New format: [Index] [Orig Start ms] [Orig End ms] [Orig Duration ms] [Synced Start ms]
        -- Example: [2] [630ms] [2790ms] [2160ms] [49879ms]
        local idx_n, s_ms, e_ms, d_ms, sy_ms = line:match("%[%s*(%d+)%s*%]%s*%[%s*(%d+)%s*ms%s*%]%s*%[%s*(%d+)%s*ms%s*%]%s*%[%s*(%d+)%s*ms%s*%]%s*%[%s*(%d+)%s*ms%s*%]")

        if idx_n then
            start_orig = ms_to_seconds(s_ms)
            end_orig   = ms_to_seconds(e_ms)
            start_sync = ms_to_seconds(sy_ms)
            duration   = ms_to_seconds(d_ms)
            -- Fallback: if duration not sensible, derive from start/end
            if not duration or duration <= 0 then
                duration = end_orig - start_orig
            end
        else
            -- Legacy format fallback: HH:MM:SS,mmm  HH:MM:SS,mmm  HH:MM:SS,mmm
            local ts1, ts2, ts3 = line:match("(%d+:%d+:%d+,%d+).-(%d+:%d+:%d+,%d+).-(%d+:%d+:%d+,%d+)")
            if ts1 and ts2 and ts3 then
                start_orig = hms_to_seconds(ts1)
                end_orig   = hms_to_seconds(ts2)
                start_sync = hms_to_seconds(ts3)
                duration   = end_orig - start_orig
            end
        end

        if start_orig and start_sync and duration then
            if duration > 0 then
                -- 1. Create a new item on the destination track
                local new_item = reaper.AddMediaItemToTrack(dest_track)
                
                -- 2. Copy the source take to the new item
                local src_take = reaper.GetActiveTake(sel_item)
                local new_take = reaper.AddTakeToMediaItem(new_item)
                
                -- Point the new take to the same source file
                local pcm_src = reaper.GetMediaItemTake_Source(src_take)
                reaper.SetMediaItemTake_Source(new_take, pcm_src)
                
                -- 3. Set Position and Offsets
                reaper.SetMediaItemInfo_Value(new_item, "D_POSITION", start_sync)
                reaper.SetMediaItemInfo_Value(new_item, "D_LENGTH", duration)
                
                -- Set the start offset (cut the segment)
                local src_offset = reaper.GetMediaItemTakeInfo_Value(src_take, "D_STARTOFFS")
                reaper.SetMediaItemTakeInfo_Value(new_take, "D_STARTOFFS", start_orig + src_offset)
            end
        end
    end

    file:close()
    reaper.UpdateArrange()
    reaper.Undo_EndBlock("Sync items from file", -1)
end

main()