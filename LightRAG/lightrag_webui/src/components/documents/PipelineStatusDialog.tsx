import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlignLeft, AlignCenter, AlignRight } from 'lucide-react'

function parseChunkProgress(messages: string[]): { current: number; total: number } | null {
  const regex = /Chunk (\d+) of (\d+) extracted/
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = regex.exec(messages[i])
    if (m) return { current: parseInt(m[1], 10), total: parseInt(m[2], 10) }
  }
  return null
}

function parseMergeProgress(messages: string[]): number {
  return messages.filter(m => /LLMmrg:/.test(m)).length
}

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}h ${m % 60}m`
  if (m > 0) return `${m}m ${s % 60}s`
  return `${s}s`
}

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import { getPipelineStatus, cancelPipeline, PipelineStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { cn } from '@/lib/utils'

type DialogPosition = 'left' | 'center' | 'right'

interface PipelineStatusDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function PipelineStatusDialog({
  open,
  onOpenChange
}: PipelineStatusDialogProps) {
  const { t } = useTranslation()
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null)
  const [position, setPosition] = useState<DialogPosition>('center')
  const [isUserScrolled, setIsUserScrolled] = useState(false)
  const [showCancelConfirm, setShowCancelConfirm] = useState(false)
  const [now, setNow] = useState(Date.now())
  const historyRef = useRef<HTMLDivElement>(null)
  const chunkSamplesRef = useRef<{n: number; t: number}[]>([])

  // Reset position when dialog opens
  useEffect(() => {
    if (open) {
      setPosition('center')
      setIsUserScrolled(false)
      chunkSamplesRef.current = []
    } else {
      // Reset confirmation dialog state when main dialog closes
      setShowCancelConfirm(false)
    }
  }, [open])

  // Handle scroll position
  useEffect(() => {
    const container = historyRef.current
    if (!container || isUserScrolled) return

    container.scrollTop = container.scrollHeight
  }, [status?.history_messages, isUserScrolled])

  const handleScroll = () => {
    const container = historyRef.current
    if (!container) return

    const isAtBottom = Math.abs(
      (container.scrollHeight - container.scrollTop) - container.clientHeight
    ) < 1

    if (isAtBottom) {
      setIsUserScrolled(false)
    } else {
      setIsUserScrolled(true)
    }
  }

  // Refresh status every 2 seconds
  useEffect(() => {
    if (!open) return

    const fetchStatus = async () => {
      try {
        const data = await getPipelineStatus()
        setStatus(data)
        // Update rolling window sample for ETA
        const msgs = data.history_messages ?? []
        for (let i = msgs.length - 1; i >= 0; i--) {
          const m = /Chunk (\d+) of \d+ extracted/.exec(msgs[i])
          if (m) {
            const n = parseInt(m[1], 10)
            const samples = chunkSamplesRef.current
            if (!samples.length || samples[samples.length - 1].n !== n) {
              chunkSamplesRef.current = [...samples.slice(-19), { n, t: Date.now() }]
            }
            break
          }
        }
      } catch (err) {
        toast.error(t('documentPanel.pipelineStatus.errors.fetchFailed', { error: errorMessage(err) }))
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    const tickInterval = setInterval(() => setNow(Date.now()), 1000)
    return () => { clearInterval(interval); clearInterval(tickInterval) }
  }, [open, t])

  // Handle cancel pipeline confirmation
  const handleConfirmCancel = async () => {
    setShowCancelConfirm(false)
    try {
      const result = await cancelPipeline()
      if (result.status === 'cancellation_requested') {
        toast.success(t('documentPanel.pipelineStatus.cancelSuccess'))
      } else if (result.status === 'not_busy') {
        toast.info(t('documentPanel.pipelineStatus.cancelNotBusy'))
      }
    } catch (err) {
      toast.error(t('documentPanel.pipelineStatus.cancelFailed', { error: errorMessage(err) }))
    }
  }

  // Determine if cancel button should be enabled
  const canCancel = status?.busy === true && !status?.cancellation_requested

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'sm:max-w-[800px] transition-all duration-200 fixed',
          position === 'left' && '!left-[25%] !translate-x-[-50%] !mx-4',
          position === 'center' && '!left-1/2 !-translate-x-1/2',
          position === 'right' && '!left-[75%] !translate-x-[-50%] !mx-4'
        )}
      >
        <DialogDescription className="sr-only">
          {status?.job_name
            ? `${t('documentPanel.pipelineStatus.jobName')}: ${status.job_name}, ${t('documentPanel.pipelineStatus.progress')}: ${status.cur_batch}/${status.batchs}`
            : t('documentPanel.pipelineStatus.noActiveJob')
          }
        </DialogDescription>
        <DialogHeader className="flex flex-row items-center">
          <DialogTitle className="flex-1">
            {t('documentPanel.pipelineStatus.title')}
          </DialogTitle>

          {/* Position control buttons */}
          <div className="flex items-center gap-2 mr-8">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'left' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('left')}
            >
              <AlignLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'center' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('center')}
            >
              <AlignCenter className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'right' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('right')}
            >
              <AlignRight className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        {/* Status Content */}
        <div className="space-y-4 pt-4">
          {/* Pipeline Status - with cancel button */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Left side: Status indicators */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.busy')}:</div>
                <div className={`h-2 w-2 rounded-full ${status?.busy ? 'bg-green-500' : 'bg-gray-300'}`} />
              </div>
              <div className="flex items-center gap-2">
                <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.requestPending')}:</div>
                <div className={`h-2 w-2 rounded-full ${status?.request_pending ? 'bg-green-500' : 'bg-gray-300'}`} />
              </div>
              {/* Only show cancellation status when it's requested */}
              {status?.cancellation_requested && (
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.cancellationRequested')}:</div>
                  <div className="h-2 w-2 rounded-full bg-red-500" />
                </div>
              )}
            </div>

            {/* Right side: Cancel button - only show when pipeline is busy */}
            {status?.busy && (
              <Button
                variant="destructive"
                size="sm"
                disabled={!canCancel}
                onClick={() => setShowCancelConfirm(true)}
                title={
                  status?.cancellation_requested
                    ? t('documentPanel.pipelineStatus.cancelInProgress')
                    : t('documentPanel.pipelineStatus.cancelTooltip')
                }
              >
                {t('documentPanel.pipelineStatus.cancelButton')}
              </Button>
            )}
          </div>

          {/* Job Information */}
          <div className="rounded-md border p-3 space-y-2">
            <div>{t('documentPanel.pipelineStatus.jobName')}: {status?.job_name || '-'}</div>
            <div className="flex justify-between">
              <span>{t('documentPanel.pipelineStatus.startTime')}: {status?.job_start
                ? new Date(status.job_start).toLocaleString(undefined, {
                  year: 'numeric',
                  month: 'numeric',
                  day: 'numeric',
                  hour: 'numeric',
                  minute: 'numeric',
                  second: 'numeric'
                })
                : '-'}</span>
              <span>{t('documentPanel.pipelineStatus.progress')}: {status ? `${status.cur_batch}/${status.batchs} ${t('documentPanel.pipelineStatus.unit')}` : '-'}</span>
            </div>
          </div>

          {/* Progress bar (shown when busy) */}
          {status?.busy && (() => {
            const chunkProg = parseChunkProgress(status.history_messages ?? [])
            const jobStart = status.job_start ? new Date(status.job_start).getTime() : null
            const elapsed = jobStart ? now - jobStart : 0
            const multiDoc = status.batchs > 1

            let pct: number | null = null
            let label = ''
            let remaining: string | null = null

            if (chunkProg) {
              pct = Math.min(99, Math.round((chunkProg.current / chunkProg.total) * 100))
              label = `Chunk ${chunkProg.current.toLocaleString()} / ${chunkProg.total.toLocaleString()} · extracting entities`
              const samples = chunkSamplesRef.current
              if (samples.length >= 2) {
                const w = samples.slice(-10)
                const msPerChunk = (w[w.length - 1].t - w[0].t) / (w[w.length - 1].n - w[0].n)
                if (msPerChunk > 0 && isFinite(msPerChunk)) {
                  remaining = formatDuration(msPerChunk * (chunkProg.total - chunkProg.current))
                }
              }
            } else if (multiDoc) {
              const done = Math.max(0, status.cur_batch - 1)
              pct = Math.round((done / status.batchs) * 100)
              label = `Document ${status.cur_batch} / ${status.batchs}`
              if (elapsed > 5000 && done > 0) {
                const msPerDoc = elapsed / done
                remaining = formatDuration(msPerDoc * (status.batchs - done))
              }
            } else {
              const mergeCount = parseMergeProgress(status.history_messages ?? [])
              const isMerging = mergeCount > 0 || (status.latest_message ?? '').includes('LLMmrg')
              label = isMerging
                ? `Merging knowledge graph · ${mergeCount.toLocaleString()} entities merged`
                : status.latest_message || 'Extracting entities…'
            }

            const indeterminate = pct === null

            return (
              <div className="space-y-1.5">
                <style>{`
                  @keyframes dlg-indeterminate {
                    0%   { transform: translateX(-100%) scaleX(0.4); }
                    50%  { transform: translateX(50%)  scaleX(0.6); }
                    100% { transform: translateX(200%) scaleX(0.4); }
                  }
                  .dlg-indeterminate { animation: dlg-indeterminate 1.6s ease-in-out infinite; }
                `}</style>
                <div className="flex justify-between text-sm">
                  <span className="font-medium">
                    {indeterminate ? label : `${label} — ${pct}%`}
                  </span>
                  <span className="text-muted-foreground text-xs">
                    {elapsed > 0 && `${formatDuration(elapsed)} elapsed`}
                    {remaining && ` · ~${remaining} left`}
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden relative">
                  {indeterminate ? (
                    <div className="absolute inset-0 bg-blue-500 h-2.5 rounded-full dlg-indeterminate" />
                  ) : (
                    <div
                      className="bg-blue-500 h-2.5 rounded-full transition-all duration-700"
                      style={{ width: `${pct}%` }}
                    />
                  )}
                </div>
              </div>
            )
          })()}

          {/* History Messages */}
          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.pipelineMessages')}:</div>
            <div
              ref={historyRef}
              onScroll={handleScroll}
              className="font-mono text-xs rounded-md bg-zinc-800 text-zinc-100 p-3 overflow-y-auto overflow-x-hidden min-h-[7.5em] max-h-[40vh]"
            >
              {status?.history_messages?.length ? (
                status.history_messages.map((msg, idx) => (
                  <div key={idx} className="whitespace-pre-wrap break-all">{msg}</div>
                ))
              ) : '-'}
            </div>
          </div>
        </div>
      </DialogContent>

      {/* Cancel Confirmation Dialog */}
      <Dialog open={showCancelConfirm} onOpenChange={setShowCancelConfirm}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>{t('documentPanel.pipelineStatus.cancelConfirmTitle')}</DialogTitle>
            <DialogDescription>
              {t('documentPanel.pipelineStatus.cancelConfirmDescription')}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-3 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowCancelConfirm(false)}
            >
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmCancel}
            >
              {t('documentPanel.pipelineStatus.cancelConfirmButton')}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Dialog>
  )
}
