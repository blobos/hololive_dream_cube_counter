#Requires AutoHotkey v2.0

; ==========================================
; CONFIGURATION SETTINGS
; ==========================================
global RunMacro := false
global MouseSpeedMultiplier := 8  ; Adjust speed here (higher = faster mouse glide)
; ==========================================

CoordMode("Mouse", "Screen")
SendMode("Event")
SetTitleMatchMode(2)

; Toggle macro loop with F1
F1:: {
    global RunMacro := !RunMacro
    if RunMacro {
        ToolTip("Macro Started")
        SetTimer(PerformMacroTasks, 10)
    } else {
        StopMacro()
    }
}

F6::ExitApp()

StopMacro() {
    global RunMacro := false
    SetTimer(PerformMacroTasks, 0)
    ToolTip("Macro Stopped")
    SetTimer(() => ToolTip(), -1000)
}

PerformMacroTasks() {
    if !RunMacro {
        SetTimer(PerformMacroTasks, 0)
        return
    }

    tt := "hololive-Dreams ahk_class UnityWndClass"
    if !WinExist(tt)
        return
    WinActivate(tt)

    ; Track loop count in registry
    StartingValue := 0
    i := RegRead("HKEY_CURRENT_USER\SOFTWARE\" A_ScriptName, "i", StartingValue)
    RegWrite(i + 1, "REG_DWORD", "HKEY_CURRENT_USER\SOFTWARE\" A_ScriptName, "i")

    ; 1. #next (Screen: 1858, 554)
    if !SmoothClick(1858, 554)
        return
    if !InterruptibleSleep(800, 1500)
        return

    ; 2. #holomemboard (Screen: 414, 564)
    if !SmoothClick(414, 564, true)
        return
    if !InterruptibleSleep(800, 1500)
        return

    ; 3. #reset (Screen: 534, 988) - 10 to 21 clicks skewed toward lower range
    if !SmoothClick(534, 988, false, true)
        return
    if !InterruptibleSleep(1000, 2000)
        return

    ; Screenshot: Windows Key + PrtScn
    Send("#{PrintScreen}")
    if !InterruptibleSleep(800, 1500)
        return

    ; 4. #exit reset (Screen: 1555, 161)
    if !SmoothClick(1555, 161)
        return
    if !InterruptibleSleep(800, 1500)
        return

    ; Keyboard: Esc
    Send("{Esc}")
    if !InterruptibleSleep(1000, 2000)
        return
}

; Interruptible sleep that checks for stop signal every 50ms
InterruptibleSleep(minTime, maxTime) {
    global RunMacro
    targetWait := Random(minTime, maxTime)
    elapsed := 0
    while (elapsed < targetWait) {
        if (!RunMacro) {
            StopMacro()
            return false
        }
        Sleep(50)
        elapsed += 50
    }
    return RunMacro
}

; Smooth motion trail function matching your screen coordinates
SmoothClick(targetX, targetY, doubleClick := false, isReset := false) {
    global RunMacro
    if (!RunMacro)
        return false

    MouseGetPos(&startX, &startY)
    
    distance := Sqrt((targetX - startX)**2 + (targetY - startY)**2)
    baseSteps := Max(Round(distance / 30), 1)
    steps := Max(Round(baseSteps / MouseSpeedMultiplier), 1)
    
    loop steps {
        if (!RunMacro)
            return false
        progress := A_Index / steps
        currentX := startX + (targetX - startX) * progress + Random(-1, 1)
        currentY := startY + (targetY - startY) * progress + Random(-1, 1)
        
        MouseMove(currentX, currentY)
    }
    
    if (!RunMacro)
        return false
    MouseMove(targetX, targetY)
    Sleep(15)
    
    if (isReset) {
        ; Skewed random distribution for 10-21 clicks (weights lower numbers higher)
        r := Random(1, 100)
        if (r <= 40)
            burstCount := Random(10, 13)      ; 40% chance: 10 to 13 clicks
        else if (r <= 80)
            burstCount := Random(14, 17)      ; 40% chance: 14 to 17 clicks
        else
            burstCount := Random(18, 21)      ; 20% chance: 18 to 21 clicks

        loop burstCount {
            if (!RunMacro)
                return false
            Click("Down")
            Sleep(Random(15, 30))
            Click("Up")
            Sleep(Random(35, 70))
        }
    } else if (doubleClick) {
        Click("D")
        Sleep(Random(40, 80))
        Click("U")
        Sleep(Random(50, 100))
        Click()
    } else {
        Click()
    }
    
    return RunMacro
}