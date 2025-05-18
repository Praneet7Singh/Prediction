import { Component } from '@angular/core';
import { Heart } from '../../interfaces/heart';
import { DataService } from '../../services/data.service';
import { NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SafeResourceUrl, DomSanitizer } from '@angular/platform-browser';
import { Router } from '@angular/router';
import {
  ApexAxisChartSeries,
  ApexChart,
  ApexXAxis,
  ApexPlotOptions,
  ApexDataLabels,
  ApexTooltip,
  ApexFill,
  NgApexchartsModule,
} from 'ng-apexcharts';

export type ChartOptions = {
  series: ApexAxisChartSeries;
  chart: ApexChart;
  xaxis: ApexXAxis;
  plotOptions: ApexPlotOptions;
  dataLabels: ApexDataLabels;
  tooltip: ApexTooltip;
  fill: ApexFill;
  colors: string[];
};

@Component({
  selector: 'app-heart',
  imports: [NgIf, FormsModule, NgApexchartsModule],
  templateUrl: './heart.component.html',
  styleUrl: './heart.component.css',
})
export class HeartComponent {
  HeartDiseaseInput: Heart = {
    age: 45,
    gender: 1,
    chest_pain: 1,
    resting_bp: 70,
    cholesterol: 80,
    fasting_blood_sugar: 0,
    resting_ecg: 1,
    max_heart_rate: 250,
    exercise_angina: 1,
    oldpeak: 1.0,
    slope: 1,
    major_vessels: 1,
  };
  Input: Heart = {
    age: '',
    gender: '',
    chest_pain: '',
    resting_bp: '',
    cholesterol: '',
    fasting_blood_sugar: '',
    resting_ecg: '',
    max_heart_rate: '',
    exercise_angina: '',
    oldpeak: '',
    slope: '',
    major_vessels: '',
  };
  chartOptions: Partial<ChartOptions> | undefined;
  Predicted_Class: any;
  Shap_Values: any;
  Explanation: any;
  waterfallImageSrc: SafeResourceUrl | undefined;
  AccuracyImageSrc: SafeResourceUrl | undefined;
  Result: any;
  Pred_Display = false;
  constructor(
    private ds: DataService,
    private sanitizer: DomSanitizer,
    private router: Router
  ) {}

  Heart_Disease_Prediction() {
    this.ds.Hrt_Pred(this.Input).subscribe((res: any) => {
      console.log(res);
      this.Result = res;
      this.Heart_Explain();
    });
  }

  Heart_Explain() {
    this.ds.Hrt_Explain(this.HeartDiseaseInput).subscribe((res: any) => {
      console.log(res);
      this.Shap_Values = res.shap_contributions;
      this.Predicted_Class = res.prediction;
      this.Explain_Function(this.Predicted_Class, this.Shap_Values);
      this.Pred_Display = true;
      const base64String = res.waterfall_plot;
      const dataUrl = `data:image/png;base64,${base64String}`;
      this.waterfallImageSrc =
        this.sanitizer.bypassSecurityTrustResourceUrl(dataUrl);
      this.chartOptions = {
        series: [
          {
            name: 'SHAP Value',
            data: this.Shap_Values.map((v: any) => v.shap_value),
          },
        ],
        chart: {
          type: 'bar',
          height: 400,
          animations: {
            enabled: true,
            speed: 800,
            animateGradually: { enabled: true, delay: 150 },
            dynamicAnimation: { enabled: true, speed: 350 },
          },
        },
        xaxis: {
          categories: this.Shap_Values.map((v: any) => v.feature),
          labels: {
            style: {
              colors: '#334155',
              fontSize: '14px',
            },
          },
        },
        plotOptions: {
          bar: {
            borderRadius: 6,
            columnWidth: '45%',
            distributed: true,
          },
        },
        dataLabels: {
          enabled: false,
        },
        tooltip: {
          theme: 'dark',
        },
        fill: {
          opacity: 0.85,
        },
        colors: [
          '#3b82f6',
          '#10b981',
          '#f59e42',
          '#ef4444',
          '#6366f1',
          '#f43f5e',
          '#14b8a6',
          '#fbbf24',
          '#a3e635',
          '#eab308',
          '#8b5cf6',
          '#f472b6',
        ],
      };
    });
  }
  Explain_Function(predictedClass: any, shapValues: any) {
    const positiveFeatures = shapValues
      .filter((f: any) => f.shap_value > 0)
      .map((f: any) => f.feature);

    const negativeFeatures = shapValues
      .filter((f: any) => f.shap_value < 0)
      .map((f: any) => f.feature);

    // Build explanation string
    let explanation = `Explanation of SHAP values affecting the prediction of class '${predictedClass}' :\n`;

    if (positiveFeatures.length > 0) {
      explanation += `Features with positive impact: ${positiveFeatures.join(
        ', '
      )}.\n`;
    } else {
      explanation += `No features with positive impact.\n`;
    }

    if (negativeFeatures.length > 0) {
      explanation += `Features with negative impact: ${negativeFeatures.join(
        ', '
      )}.\n`;
    } else {
      explanation += `No features with negative impact.\n`;
    }

    this.Explanation = explanation;
    console.log(this.Explanation);
  }
}
