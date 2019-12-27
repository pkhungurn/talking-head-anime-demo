import math

LEFT_EYE_HORIZ_POINTS = [36, 39]
LEFT_EYE_TOP_POINTS = [37, 38]
LEFT_EYE_BOTTOM_POINTS = [41, 40]

RIGHT_EYE_HORIZ_POINTS = [42, 45]
RIGHT_EYE_TOP_POINTS = [43, 44]
RIGHT_EYE_BOTTOM_POINTS = [47, 46]


def compute_left_eye_normalized_ratio(face_landmarks, min_ratio, max_ratio):
    return compute_eye_normalized_ratio(face_landmarks,
                                        LEFT_EYE_HORIZ_POINTS, LEFT_EYE_BOTTOM_POINTS, LEFT_EYE_TOP_POINTS,
                                        min_ratio, max_ratio)


def compute_right_eye_normalized_ratio(face_landmarks, min_ratio, max_ratio):
    return compute_eye_normalized_ratio(face_landmarks,
                                        RIGHT_EYE_HORIZ_POINTS, RIGHT_EYE_BOTTOM_POINTS, RIGHT_EYE_TOP_POINTS,
                                        min_ratio, max_ratio)


def compute_eye_normalized_ratio(face_landmarks, eye_horiz_points, eye_bottom_points, eye_top_points, min_ratio,
                                 max_ratio):
    left_eye_horiz_diff = face_landmarks.part(eye_horiz_points[0]) - face_landmarks.part(eye_horiz_points[1])
    left_eye_horiz_length = math.sqrt(left_eye_horiz_diff.x ** 2 + left_eye_horiz_diff.y ** 2)
    left_eye_top_point = (face_landmarks.part(eye_top_points[0]) + face_landmarks.part(eye_top_points[1])) / 2.0
    left_eye_bottom_point = (face_landmarks.part(eye_bottom_points[0]) + face_landmarks.part(
        eye_bottom_points[1])) / 2.0
    left_eye_vert_diff = left_eye_top_point - left_eye_bottom_point
    left_eye_vert_length = math.sqrt(left_eye_vert_diff.x ** 2 + left_eye_vert_diff.y ** 2)
    left_eye_ratio = left_eye_vert_length / left_eye_horiz_length
    left_eye_normalized_ratio = (min(max(left_eye_ratio, min_ratio), max_ratio) - min_ratio) / (
            max_ratio - min_ratio)
    return left_eye_normalized_ratio


MOUTH_TOP_POINTS = [61, 62, 63]
MOUTH_BOTTOM_POINTS = [67, 66, 65]
MOUTH_HORIZ_POINTS = [60, 64]


def compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio):
    mouth_top_point = (face_landmarks.part(MOUTH_TOP_POINTS[0])
                       + face_landmarks.part(MOUTH_TOP_POINTS[1])
                       + face_landmarks.part(MOUTH_TOP_POINTS[2])) / 3.0
    mouth_bottom_point = (face_landmarks.part(MOUTH_BOTTOM_POINTS[0])
                          + face_landmarks.part(MOUTH_BOTTOM_POINTS[1])
                          + face_landmarks.part(MOUTH_BOTTOM_POINTS[2])) / 3.0
    mouth_vert_diff = mouth_top_point - mouth_bottom_point
    mouth_vert_length = math.sqrt(mouth_vert_diff.x ** 2 + mouth_vert_diff.y ** 2)
    mouth_horiz_diff = face_landmarks.part(MOUTH_HORIZ_POINTS[0]) - face_landmarks.part(MOUTH_HORIZ_POINTS[1])
    mouth_horiz_length = math.sqrt(mouth_horiz_diff.x ** 2 + mouth_horiz_diff.y ** 2)
    mouth_ratio = mouth_vert_length / mouth_horiz_length
    mouth_normalized_ratio = (min(max(mouth_ratio, min_mouth_ratio), max_mouth_ratio) - min_mouth_ratio) / (
            max_mouth_ratio - min_mouth_ratio)
    return mouth_normalized_ratio
