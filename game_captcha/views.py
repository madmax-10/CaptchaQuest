from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import CaptchaSession, MouseMovement
import uuid
import numpy as np
import math
from scipy.stats import entropy
import requests
import os

def success_view(request):
    # This is where users will be redirected after clicking "Continue"
    continue_url = request.session.get('continue_url', '/captcha')
    return render(request, 'game_captcha/success.html', {'continue_url': continue_url})

# OpenRouter configuration
OPENROUTER_API_KEY = None
with open("api.txt", "r") as apiFile:
    OPENROUTER_API_KEY = apiFile.read().strip()
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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

def classify_movement_with_ai(metrics, movements):
    """
    Use DeepSeek AI through OpenRouter to analyze the movement patterns
    and provide a more sophisticated human vs bot detection with improved
    sensitivity to human-like movements
    """
    if not OPENROUTER_API_KEY:
        # Fall back to traditional algorithm if API key is not available
        return classify_movement(metrics, human_bias=True)
    
    # Prepare data for AI analysis with better sampling
    print("Using AI with improved human detection")
    
    # Sample more efficiently - keep beginning, middle and end points
    movement_summary = []
    if len(movements) <= 100:
        # If we have fewer than 100 points, use all of them
        movement_summary = movements
    else:
        # Take first 20 points (beginning of movement)
        movement_summary.extend(movements[:20])
        
        # Take middle points
        middle_start = len(movements) // 3
        middle_end = 2 * len(movements) // 3
        middle_step = max(1, (middle_end - middle_start) // 30)
        movement_summary.extend(movements[middle_start:middle_end:middle_step])
        
        # Take last 20 points (end of movement)
        movement_summary.extend(movements[-20:])
    
    # Format key metrics for AI analysis
    metrics_summary = {
        'speed_variance_ratio': metrics.get('speed_variance_ratio', 0),
        'acceleration_changes_ratio': metrics.get('acceleration_changes_ratio', 0),
        'direction_changes_ratio': metrics.get('direction_changes_ratio', 0),
        'path_efficiency': metrics.get('path_efficiency', 0),
        'speed_entropy': metrics.get('speed_entropy', 0),
        'jerk_ratio': metrics.get('jerk_ratio', 0),
        'pause_ratio': metrics.get('pause_ratio', 0),
        'hover_ratio': metrics.get('hover_ratio', 0)
    }
    
    prompt = f"""
    I need to determine if mouse movement patterns are from a human or an AI/bot.
    This analysis should lean toward classifying movements as human when in doubt.
    
    Key metrics from the movement data:
    {json.dumps(metrics_summary, indent=2)}
    
    Sample of movement data points (x, y, timestamp):
    {json.dumps([{'x': m['x'], 'y': m['y'], 'timestamp': m['timestamp']} for m in movement_summary[:15]], indent=2)}
    
    Analyze these patterns and determine if this is likely human or bot behavior.
    
    Human movements typically have these characteristics (with typical ranges):
    1. Speed variance ratio: > 0.2 (higher variance is more human-like)
    2. Acceleration changes ratio: > 0.1 (humans change acceleration frequently)
    3. Direction changes ratio: > 0.1 (humans change directions frequently)
    4. Path efficiency: < 0.9 (less efficient paths are typically human)
    5. Speed entropy: > 0.2 (diverse speed distribution is human-like)
    6. Jerk ratio: > 0.1 (some jerkiness is natural in human movement)
    7. Pause ratio: > 0.05 (humans naturally pause)
    8. Hover ratio: > 0.05 (humans hover over elements)
    
    IMPORTANT: Many humans use trackpads, different pointing devices, or have varying levels 
    of motor control, which can result in movement patterns that might initially appear bot-like.
    
    Even if only some metrics are in the human range, prefer classifying as human rather than bot.
    
    IF IN DOUBT, CLASSIFY AS HUMAN.
    
    Respond in this exact JSON format **ONLY**:
    {{
        "is_human": true/false,
        "confidence": 0.0-1.0,
        "reason": "Explanation for the decision",
        "score": 0-100
    }}
    Do NOT provide explanations or extra text. Only return a JSON object.
    """
    
    try:
        # Make API request to OpenRouter's DeepSeek model
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-r1-distill-llama-70b:free",  # Using the free DeepSeek model
            "messages": [
                {"role": "system", "content": "You are an AI trained to analyze mouse movement patterns and detect if they are from a human or a bot. Always err on the side of classifying as human when uncertain."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            ai_output = response_data['choices'][0]['message']['content']
            print(f"AI Output: {ai_output}")
            
            try:
                # Try to extract JSON from the response with improved parsing
                start_idx = ai_output.find('{')
                end_idx = ai_output.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_output = ai_output[start_idx:end_idx]
                    ai_result = json.loads(json_output)
                    
                    # Ensure all required fields are present
                    if 'is_human' in ai_result and 'confidence' in ai_result and 'reason' in ai_result:
                        # Apply confidence threshold adjustment - bias toward human classification
                        if not ai_result['is_human'] and ai_result['confidence'] < 0.85:
                            print("Bot detection below high confidence threshold, reclassifying as human")
                            ai_result['is_human'] = True
                            ai_result['confidence'] = 1.0 - ai_result['confidence']
                            ai_result['reason'] = f"Reclassified as human due to uncertainty. Original reason: {ai_result['reason']}"
                            ai_result['score'] = 100 - ai_result.get('score', 50)
                            
                        return {
                            'is_human': bool(ai_result['is_human']),
                            'confidence': float(ai_result['confidence']),
                            'reason': ai_result['reason'],
                            'score': float(ai_result.get('score', 50.0)),
                            'detailed_metrics': metrics,
                            'ai_powered': True
                        }
            except json.JSONDecodeError:
                print("Failed to parse AI response as JSON")
            except Exception as e:
                print(f"Error processing AI response: {str(e)}")
        
        # Fall back to traditional algorithm if AI fails
        print("Falling back to traditional algorithm due to AI response failure")
        traditional_result = classify_movement(metrics, human_bias=True)
        traditional_result['ai_powered'] = False
        return traditional_result
        
    except Exception as e:
        print(f"Error using AI for classification: {str(e)}")
        # Fall back to traditional algorithm on exception
        traditional_result = classify_movement(metrics, human_bias=True)
        traditional_result['ai_powered'] = False
        return traditional_result


def classify_movement(metrics, human_bias=False):
    """
    Use the calculated metrics to determine if movement is human-like
    with option to bias toward human classification
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
            'is_human': human_bias,  # Default to human if human_bias is True
            'confidence': 0.5,
            'reason': 'No valid metrics available',
            'score': 50 if human_bias else 0,
            'ai_powered': False
        }
    
    # Calculate weighted average with adjustments for human bias
    # If human_bias is True, we'll lower the weights of metrics that might falsely indicate bots
    if human_bias:
        # Emphasize metrics that are more reliable for human detection
        weights = [0.25, 0.15, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05]  # Adjusted weights
    else:
        # Original weights
        weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    valid_weights = weights[:len(valid_scores)]
    
    # Normalize weights
    weight_sum = sum(valid_weights)
    if weight_sum > 0:
        valid_weights = [w / weight_sum for w in valid_weights]
    
    # Calculate total score
    total_score = sum(s * w for s, w in zip(valid_scores, valid_weights))
    
    # Determine if human based on threshold
    # Lower the threshold if human_bias is True
    threshold = 30 if human_bias else 40
    is_human = total_score >= threshold
    
    # Human bias override: If any individual score is very high, consider it human
    if human_bias and not is_human:
        # If any individual metric is strongly indicative of human movement
        max_score = max(valid_scores)
        if max_score > 70:  # High score in any one metric suggests human
            is_human = True
            total_score = max(total_score, 50)  # Ensure score is at least 50
    
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
    
    result = {
        'is_human': is_human,
        'score': total_score,
        'confidence': confidence,
        'reason': f"{'Human-like' if is_human else 'Bot-like'} movement detected: {primary_reason}",
        'detailed_metrics': metrics,
        'ai_powered': False
    }
    
    print(result)
    return result


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
            
            # Use the AI-powered classification if movements are sufficient
            if len(movements) >= 10:
                result = classify_movement_with_ai(metrics, movements)
            else:
                result = {
                    'is_human': False, 
                    'confidence': 1.0, 
                    'reason': 'Insufficient movement data for analysis',
                    'score': 0,
                    'ai_powered': False
                }
                
            is_human = bool(result['is_human'])  # Convert to native Python bool
            
            # Update session status
            if game_completed:
                session.completed = True
                session.passed = game_success and is_human
                session.save()
            
            response_data = {
                'success': bool(session.passed),  # Convert to native Python bool
                'is_human': is_human,
                'message': result['reason'],
                'ai_powered': bool(result.get('ai_powered', False))
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