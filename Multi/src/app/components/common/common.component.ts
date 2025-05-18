import { Component } from '@angular/core';
import { DataService } from '../../services/data.service';
import { FormsModule } from '@angular/forms';
import { NgFor, NgIf } from '@angular/common';
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
  selector: 'app-common',
  imports: [FormsModule, NgIf, NgFor, NgApexchartsModule],
  templateUrl: './common.component.html',
  styleUrl: './common.component.css',
})
export class CommonComponent {
  TextInput = { symptoms: '' };
  Result: any;
  Explanation: any;
  Shap_Values: any;
  Probability: any;
  chartOptions: Partial<ChartOptions> | undefined;
  Pred_Display = false;
  constructor(private ds: DataService) {}
  ngOnInit() {}
  Sym_Predict() {
    this.ds.Sym_Pred(this.TextInput).subscribe((res: any) => {
      console.log(res);
      this.Result = res.disease;
      this.Probability = res.probability;
      this.Pred_Display = true;
    });
    this.Sym_Explain();
  }
  Sym_Explain() {
    this.ds.Sym_Explain(this.TextInput).subscribe((res: any) => {
      console.log(res);
      this.Shap_Values = res.top_features;
      this.Explain_Function(this.Result, this.Shap_Values);
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
