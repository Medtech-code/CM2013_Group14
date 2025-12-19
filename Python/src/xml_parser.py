import xml.etree.ElementTree as ET
import numpy as np


def parse_xml_annotations(xml_file_path):

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML file {xml_file_path}: {e}")

    epoch_length = 30
    epoch_elements = root.findall('.//EpochLength')
    if epoch_elements:
        epoch_length = float(epoch_elements[0].text)

    events = []
    stages = []

    stage_map = {
        'SDO:NonRapidEyeMovementSleep-N1': 1,  
        'SDO:NonRapidEyeMovementSleep-N2': 2,  
        'SDO:NonRapidEyeMovementSleep-N3': 3,  
        'SDO:NonRapidEyeMovementSleep-N4': 3, 
        'SDO:RapidEyeMovementSleep': 4,        
        'SDO:WakeState': 0                    
    }

    scored_events = root.findall('.//ScoredEvent')

    for event in scored_events:
        concept_elem = event.find('EventConcept')
        if concept_elem is None:
            continue

        concept = concept_elem.text.strip()
        start_elem = event.find('Start')
        duration_elem = event.find('Duration')

        if start_elem is None or duration_elem is None:
            continue

        start_time = float(start_elem.text)
        duration = float(duration_elem.text)
        event_dict = {
            'concept': concept,
            'start': start_time,
            'duration': duration
        }
        if event.find('Desaturation') is not None:
            event_dict['desaturation'] = float(event.find('Desaturation').text)
        if event.find('SpO2Nadir') is not None:
            event_dict['spo2_nadir'] = float(event.find('SpO2Nadir').text)
        if event.find('Text') is not None:
            event_dict['text'] = event.find('Text').text

        events.append(event_dict)
        if concept in stage_map:
            stage_event = {
                'stage': stage_map[concept],
                'start': start_time,
                'duration': duration
            }
            stages.append(stage_event)

    return {
        'events': events,
        'stages': stages,
        'epoch_length': epoch_length
    }


def create_epoch_labels(stages, total_duration, epoch_length=30):

    n_epochs = int(np.ceil(total_duration / epoch_length))
    labels = np.zeros(n_epochs, dtype=int)

    for stage_event in stages:
        stage = stage_event['stage']
        start = stage_event['start']
        duration = stage_event['duration']

        start_epoch = int(start / epoch_length)
        end_time = start + duration
        end_epoch = int(np.ceil(end_time / epoch_length))
        labels[start_epoch:end_epoch] = stage

    return labels


def validate_annotations(xml_file_path, edf_duration):

    parsed = parse_xml_annotations(xml_file_path)
    stages = parsed['stages']

    if not stages:
        return {
            'valid': False,
            'annotation_duration': 0,
            'coverage': 0,
            'gaps': [],
            'overlaps': [],
            'message': 'No sleep stage annotations found'
        }

    stages = sorted(stages, key=lambda x: x['start'])
    last_event = stages[-1]
    annotation_duration = last_event['start'] + last_event['duration']
    gaps = []
    for i in range(len(stages) - 1):
        current_end = stages[i]['start'] + stages[i]['duration']
        next_start = stages[i + 1]['start']

        if next_start > current_end:
            gaps.append({
                'start': current_end,
                'end': next_start,
                'duration': next_start - current_end
            })

    overlaps = []
    for i in range(len(stages) - 1):
        current_end = stages[i]['start'] + stages[i]['duration']
        next_start = stages[i + 1]['start']

        if next_start < current_end:
            overlaps.append({
                'event1_end': current_end,
                'event2_start': next_start,
                'overlap_duration': current_end - next_start
            })
    coverage = (annotation_duration / edf_duration) * 100 if edf_duration > 0 else 0

    return {
        'valid': len(gaps) == 0 and len(overlaps) == 0 and coverage >= 99,
        'annotation_duration': annotation_duration,
        'edf_duration': edf_duration,
        'coverage': coverage,
        'gaps': gaps,
        'overlaps': overlaps,
        'n_stages': len(stages)
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        xml_file = sys.argv[1]

        print(f"Parsing XML file: {xml_file}")
        parsed = parse_xml_annotations(xml_file)

        print(f"\nEpoch length: {parsed['epoch_length']} seconds")
        print(f"Total stage events: {len(parsed['stages'])}")
        print(f"Total events (all types): {len(parsed['events'])}")

        if parsed['stages']:
            stages_array = np.array([s['stage'] for s in parsed['stages']])
            stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

            print("\nStage distribution:")
            for stage_num in range(5):
                count = np.sum(stages_array == stage_num)
                if count > 0:
                    pct = (count / len(stages_array)) * 100
                    print(f"  {stage_names[stage_num]}: {count} events ({pct:.1f}%)")
            total_duration = sum(s['duration'] for s in parsed['stages'])
            print(f"\nTotal annotated duration: {total_duration:.0f} seconds ({total_duration/3600:.2f} hours)")
    else:
        print("Usage: python xml_parser.py <xml_file_path>")
