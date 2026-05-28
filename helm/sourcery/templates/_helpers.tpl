{{/* Shared template helpers for the sourcery chart. */}}

{{- define "sourcery.fullname" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "sourcery.labels" -}}
app.kubernetes.io/name: {{ include "sourcery.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "sourcery.backend.image" -}}
{{- $tag := .Values.image.backend.tag | default .Chart.AppVersion -}}
{{ .Values.image.backend.repository }}:{{ $tag }}
{{- end -}}

{{- define "sourcery.frontend.image" -}}
{{- $tag := .Values.image.frontend.tag | default .Chart.AppVersion -}}
{{ .Values.image.frontend.repository }}:{{ $tag }}
{{- end -}}
