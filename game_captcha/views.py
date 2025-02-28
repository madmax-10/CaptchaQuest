from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import CaptchaSession, MouseMovement
import uuid
import numpy as np
import math
from scipy.stats import entropy

def game_view(request):
    # Create a new CAPTCHA session
    session = CaptchaSession.objects.create()
    return render(request, 'game_captcha/game.html', {'session_id': session.session_id})

def calculate_metrics(movements):
    """
    Calculate various metrics from mouse movements to detect bot behavior
    """
    if len(movements) < 10:  # Need sufficient data points
        return {
            'is_human': False,
            'confidence': 0.0,
            'reason': 'Insufficient data points'
        }
    
    # Extract coordinates and timestamps
    points = [(move['x'], move['y'], move['timestamp']) for move in movements]
    
    # Calculate speeds, accelerations, and direction changes
    speeds = []
    accelerations = []
    angles = []
    jerk = []  # Rate of change of acceleration
    
    for i in range(1, len(points)):
        # Current and previous points
        x1, y1, t1 = points[i-1]
        x2, y2, t2 = points[i]
        
        # Time difference
        dt = t2 - t1
        if dt <= 0:  # Avoid division by zero
            continue
            
        # Distance and speed
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / dt
        speeds.append(speed)
        
        # Acceleration (change in speed)
        if i > 1:
            prev_speed = speeds[-2] if len(speeds) > 1 else 0
            acceleration = (speed - prev_speed) / dt
            accelerations.append(acceleration)
            
            # Jerk (change in acceleration)
            if i > 2:
                prev_acceleration = accelerations[-2] if len(accelerations) > 1 else 0
                jerk_value = (acceleration - prev_acceleration) / dt
                jerk.append(jerk_value)
        
        # Angle changes (direction changes)
        if i > 1:
            prev_x, prev_y = points[i-2][0], points[i-2][1]
            
            # Calculate vectors
            vector1 = (x1 - prev_x, y1 - prev_y)
            vector2 = (x2 - x1, y2 - y1)
            
            # Calculate angle between vectors (if they have magnitude)
            mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
                cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
                angle = math.acos(cos_angle) * 180 / math.pi
                angles.append(angle)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Speed variability (humans have variable speeds)
    if speeds:
        metrics['speed_mean'] = np.mean(speeds)
        metrics['speed_std'] = np.std(speeds)
        metrics['speed_variance_ratio'] = metrics['speed_std'] / metrics['speed_mean'] if metrics['speed_mean'] > 0 else 0
    
    # 2. Acceleration patterns (humans have variable acceleration)
    if accelerations:
        metrics['acceleration_mean'] = np.mean(accelerations)
        metrics['acceleration_std'] = np.std(accelerations)
        metrics['acceleration_changes'] = sum(1 for i in range(1, len(accelerations)) 
                                           if (accelerations[i] > 0 and accelerations[i-1] < 0) 
                                           or (accelerations[i] < 0 and accelerations[i-1] > 0))
        metrics['acceleration_changes_ratio'] = metrics['acceleration_changes'] / len(accelerations) if accelerations else 0
    
    # 3. Direction changes (humans make natural curves)
    if angles:
        metrics['angle_mean'] = np.mean(angles)
        metrics['angle_std'] = np.std(angles)
        # Count significant direction changes (> 20 degrees)
        metrics['direction_changes'] = sum(1 for angle in angles if angle > 20)
        metrics['direction_changes_ratio'] = metrics['direction_changes'] / len(angles) if angles else 0
    
    # 4. Path efficiency (direct vs actual path)
    start_x, start_y = movements[0]['x'], movements[0]['y']
    end_x, end_y = movements[-1]['x'], movements[-1]['y']
    
    # Direct distance
    direct_distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # Actual path distance
    actual_distance = 0
    prev_x, prev_y = start_x, start_y
    for move in movements[1:]:
        segment = math.sqrt((move['x'] - prev_x)**2 + (move['y'] - prev_y)**2)
        actual_distance += segment
        prev_x, prev_y = move['x'], move['y']
    
    metrics['path_efficiency'] = direct_distance / actual_distance if actual_distance > 0 else 1.0
    
    # 5. Jerk analysis (rate of change of acceleration - humans are jerky)
    if jerk:
        metrics['jerk_mean'] = np.mean(jerk)
        metrics['jerk_std'] = np.std(jerk)
        metrics['jerk_ratio'] = metrics['jerk_std'] / abs(metrics['jerk_mean']) if metrics['jerk_mean'] != 0 else 0
    
    # 6. Speed histogram entropy (humans have more varied speed distributions)
    if speeds:
        hist, _ = np.histogram(speeds, bins=10, density=True)
        metrics['speed_entropy'] = entropy(hist + 1e-10)  # Add small value to avoid log(0)
    
    # 7. Pause detection (humans often pause)
    pauses = 0
    for i in range(1, len(speeds)):
        if speeds[i] < 0.1 * np.mean(speeds):  # Speed < 10% of average
            pauses += 1
    metrics['pause_ratio'] = pauses / len(speeds) if speeds else 0
    
    # 8. Hovering behavior (humans hover near obstacles)
    hover_points = 0
    for i in range(1, len(movements)-1):
        # If speed is low but not zero, could be hovering
        if 0 < speeds[i] < 0.2 * np.mean(speeds):
            hover_points += 1
    metrics['hover_ratio'] = hover_points / len(movements) if movements else 0
    
    return metrics

def classify_movement(metrics):
    """
    Use the calculated metrics to determine if movement is human-like
    """
    # Set default values if metrics are missing
    speed_var_ratio = metrics.get('speed_variance_ratio', 0)
    accel_changes_ratio = metrics.get('acceleration_changes_ratio', 0)
    direction_changes_ratio = metrics.get('direction_changes_ratio', 0)
    path_efficiency = metrics.get('path_efficiency', 1.0)
    speed_entropy = metrics.get('speed_entropy', 0)
    jerk_ratio = metrics.get('jerk_ratio', 0)
    pause_ratio = metrics.get('pause_ratio', 0)
    hover_ratio = metrics.get('hover_ratio', 0)
    
    # Human characteristics:
    # 1. Variable speed (high variance)
    # 2. Many acceleration sign changes
    # 3. Many direction changes
    # 4. Lower path efficiency (not taking direct routes)
    # 5. High entropy in speed distribution
    # 6. High jerk ratio (uneven changes in acceleration)
    # 7. Some pauses during movement
    # 8. Some hovering behavior near obstacles
    
    # Calculate human-likeness score (0-100)
    scores = [
        min(100, speed_var_ratio * 100),  # Speed variability (0-100)
        min(100, accel_changes_ratio * 100),  # Acceleration changes (0-100)
        min(100, direction_changes_ratio * 50),  # Direction changes (0-100)
        min(100, (1 - path_efficiency) * 100),  # Inefficient path (0-100)
        min(100, speed_entropy * 50),  # Speed entropy (0-100)
        min(100, jerk_ratio * 50),  # Jerk ratio (0-100)
        min(100, pause_ratio * 200),  # Pauses (0-100)
        min(100, hover_ratio * 200)  # Hovering (0-100)
    ]
    
    # Filter out None values
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return {
            'is_human': False,
            'confidence': 0.0,
            'reason': 'No valid metrics available'
        }
    
    # Calculate weighted average
    weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]  # Weights should sum to 1
    valid_weights = weights[:len(valid_scores)]
    
    # Normalize weights
    weight_sum = sum(valid_weights)
    if weight_sum > 0:
        valid_weights = [w / weight_sum for w in valid_weights]
    
    # Calculate total score
    total_score = sum(s * w for s, w in zip(valid_scores, valid_weights))
    
    # Determine if human based on threshold
    threshold = 40  # This can be adjusted based on testing
    is_human = total_score >= threshold
    
    # Calculate confidence
    if total_score <= threshold:
        confidence = 1.0 - (total_score / threshold)
    else:
        confidence = (total_score - threshold) / (100 - threshold)
    
    confidence = min(1.0, max(0.0, confidence))
    
    # Determine primary reason
    max_score_index = scores.index(max(scores))
    reasons = [
        "highly variable speed",
        "frequent acceleration changes",
        "many direction changes",
        "inefficient movement path",
        "diverse speed distribution",
        "jerky movement patterns",
        "natural pauses in movement",
        "hovering behavior near obstacles"
    ]
    
    primary_reason = reasons[max_score_index]
    
    return {
        'is_human': is_human,
        'score': total_score,
        'confidence': confidence,
        'reason': f"{'Human-like' if is_human else 'Bot-like'} movement detected: {primary_reason}",
        'detailed_metrics': metrics
    }


@csrf_exempt
def verify_captcha(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        movements = data.get('movements', [])
        game_completed = data.get('completed', False)
        game_success = data.get('success', False)
        
        # For debugging - get detailed analysis
        detailed_analysis = data.get('detailed', False)
        
        try:
            session = CaptchaSession.objects.get(session_id=session_id)
            
            # Store mouse movements
            for move in movements:
                MouseMovement.objects.create(
                    session=session,
                    timestamp=move['timestamp'],
                    x=move['x'],
                    y=move['y']
                )
            
            # Bot detection algorithm
            metrics = calculate_metrics(movements)
            result = classify_movement(metrics)
            is_human = bool(result['is_human'])  # Convert to native Python bool
            
            # Update session status
            if game_completed:
                session.completed = True
                session.passed = game_success and is_human
                session.save()
            
            response_data = {
                'success': bool(session.passed),  # Convert to native Python bool
                'is_human': is_human,
                'message': result['reason']
            }
            
            # Include detailed metrics if requested
            if detailed_analysis:
                # Convert NumPy types to native Python types
                sanitized_metrics = convert_numpy_to_python(metrics)
                response_data['metrics'] = sanitized_metrics
                response_data['score'] = float(result['score'])  # Convert to native Python float
                response_data['confidence'] = float(result['confidence'])  # Convert to native Python float
            
            return JsonResponse(response_data)
            
        except CaptchaSession.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Invalid session.'}, status=400)
    
    return JsonResponse({'success': False, 'message': 'Invalid request.'}, status=400)

def convert_numpy_to_python(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj