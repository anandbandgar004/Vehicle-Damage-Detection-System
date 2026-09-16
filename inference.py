# inference.py
import argparse
import glob
import os
from pathlib import Path
import json
from damage_detector import VehicleDamageDetector

def main():
    parser = argparse.ArgumentParser(description='Run inference on vehicle images')
    parser.add_argument('--model', type=str, default='runs/train/damage_detection/weights/best.pt',
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--images', type=str, default='sample_images',
                       help='Path to image or directory of images')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - INFERENCE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\n❌ Model file not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python train_model.py")
        print("\nOr if you have a trained model, specify the correct path.")
        return
    
    # Get image paths
    if os.path.isdir(args.images):
        image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.images, ext)))
    else:
        image_paths = [args.images]
    
    if not image_paths:
        print(f"\n❌ No images found at {args.images}")
        print("\nPlease add some vehicle images to the directory.")
        print("Supported formats: .jpg, .jpeg, .png")
        return
    
    print(f"\nFound {len(image_paths)} images:")
    for img_path in image_paths[:10]:  # Show first 10
        print(f"  • {os.path.basename(img_path)}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    print(f"\nLoading model: {os.path.basename(args.model)}")
    detector = VehicleDamageDetector(model_path=args.model, device='cpu')
    
    # Run analysis
    print(f"\nAnalyzing images...")
    report = detector.analyze_multiple_images(
        image_paths=image_paths,
        confidence_threshold=args.conf
    )
    
    # Print results
    print("\n" + "="*60)
    print("DAMAGE ASSESSMENT RESULTS")
    print("="*60)
    
    print(f"\n📊 SUMMARY")
    print(f"  Images analyzed: {report['summary']['total_images_analyzed']}")
    print(f"  Total damages detected: {report['summary']['total_damages_detected']}")
    print(f"  Overall severity: {report['summary']['overall_severity']}")
    
    if report['damage_breakdown']['by_type']:
        print(f"\n🔍 DAMAGE TYPES:")
        for damage_type, count in report['damage_breakdown']['by_type'].items():
            print(f"  {damage_type}: {count}")
    
    print(f"\n⚠️  SEVERITY DISTRIBUTION:")
    for severity, count in report['damage_breakdown']['by_severity'].items():
        print(f"  {severity}: {count}")
    
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"\n✅ No damages detected - vehicle appears to be in good condition!")
    
    # Save report
    report_file = os.path.join(args.output, 'damage_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()