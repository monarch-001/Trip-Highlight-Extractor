try:
    import mediapipe as mp
    print(\"Has solutions:\", hasattr(mp, \"solutions\"))
    import mediapipe.python.solutions
    print(\"Sub-module import OK\")
except Exception as e:
    print(\"Error:\", e)
