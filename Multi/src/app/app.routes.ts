import { Routes } from '@angular/router';
import { MainComponent } from './components/main/main.component';
import { HeartComponent } from './components/heart/heart.component';
import { DiabetesComponent } from './components/diabetes/diabetes.component';
import { PageComponent } from './components/page/page.component';
import { AlzheimerComponent } from './components/alzheimer/alzheimer.component';
import { CommonComponent } from './components/common/common.component';
import { WelcomeComponent } from './components/welcome/welcome.component';

export const routes: Routes = [
  { path: '', component: PageComponent },
  {
    path: 'main',
    component: MainComponent,
    children: [
      { path: 'heart', component: HeartComponent },
      { path: 'diabetes', component: DiabetesComponent },
      { path: 'alzheimer', component: AlzheimerComponent },
      { path: 'common', component: CommonComponent },
      { path: 'welcome', component: WelcomeComponent },
    ],
  },
];
